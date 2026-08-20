import {
  type DepthCameraFrame,
  type DepthCameraResolution,
  DepthCameraSession,
} from './camera-session.ts';
import { SourceChoice, SourceChooser } from './chooser.ts';
import {
  DEFAULT_INFERENCE_RESOLUTION,
  type InferenceResolution,
} from './depth/capture.ts';
import { createDepthSource, type DepthSource } from './depth/depth-source.ts';
import {
  defaultDevice,
  type DepthDevice,
  type ModelSize,
} from './depth/model.ts';
import { RangeStabilizer } from './depth/percentile.ts';
import { createGui, type FacingMode, type GuiHandles } from './gui.ts';
import { setupLightInput } from './light-input.ts';
import { IDENTITY_MAT2 } from './mat2.ts';
import { DepthRelightingRenderer, type RenderFrame } from './renderer.ts';

const CAMERA_FRAME_RATE = 60;
// Resolved against Vite's base so the demo photo loads from a subpath deploy too.
const DEMO_IMAGE_URL = `${import.meta.env.BASE_URL}demo.jpg`;

const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const video = document.querySelector('video') as HTMLVideoElement;
const status = document.querySelector('.status') as HTMLDivElement;
const statusMessage = document.querySelector(
  '.status-message',
) as HTMLParagraphElement;
const listenerController = new AbortController();

let renderer: DepthRelightingRenderer | undefined;
let depthSource: DepthSource | undefined;
/**
 * The device the renderer was built on, when the depth backend supplied one. Held so a
 * later model swap can tell whether it is still sharing that device.
 */
let rendererDevice: GPUDevice | undefined;
let chooser: SourceChooser | undefined;
let guiHandles: GuiHandles | undefined;
let demoImage: ImageBitmap | undefined;
/** Long side (px) of the depth-inference input; the GUI's "depth res" dial. */
let inferenceResolution: InferenceResolution = DEFAULT_INFERENCE_RESOLUTION;
/** ONNX execution provider the depth model runs on; the GUI's "engine" dropdown. */
let inferenceDevice: DepthDevice = defaultDevice();
/** EMA of the backend's per-frame inference time, shown in the GUI stats row. */
let inferenceMsEma: number | undefined;
/**
 * Bumped to stop the running render loop (`startRenderLoop`); the loop's rAF callback
 * exits as soon as it sees a newer generation.
 */
let renderLoopGeneration = 0;
/**
 * What the render loop draws each display refresh. For the camera this is refreshed
 * by `DepthCameraSession.onFrame` on every *video* frame; for a photo it is set once.
 * Undefined while no source is active.
 */
let currentFrame: RenderFrame | undefined;
/**
 * Set when `currentFrame` holds content the depth network hasn't seen: every camera
 * frame, a photo once plus again after a model or resolution change. The loop consumes
 * it as soon as no inference is in flight, so inference runs on the freshest frame
 * rather than whichever arrived when it went idle.
 */
let inferencePending = false;
let inferenceBusy = false;
/**
 * GUI "lock to depth": draw only when something changed (`renderRequested`), chiefly a
 * new depth map, instead of every display refresh. Hands the GPU time a 120 Hz relight
 * would burn to the depth network.
 */
let lockRenderToDepth = false;
let renderRequested = true;
/** Mirrors the GUI's "auto orbit"; while on, the light is driven by the orbit, not the user. */
let autoOrbitEnabled = false;
/**
 * Bumped whenever the source/facing changes (i.e. whenever `rangeStabilizer` and the
 * renderer's history are reset), so an in-flight inference against the old source is
 * dropped instead of polluting the new one.
 */
let inferenceGeneration = 0;
let starting = false;

const rangeStabilizer = new RangeStabilizer();

/** Interval between panel refreshes of the render stats row (it's a DOM write). */
const RENDER_STATS_INTERVAL_MS = 250;
let frameIntervalEma: number | undefined;
let lastFrameTime: number | undefined;
let lastRenderStatsUpdate = 0;

/** Feeds the panel's "render" row: presented frame rate + the relight pass's GPU time. */
function recordRenderedFrame(time: number, gpuMs: number | undefined): void {
  if (lastFrameTime !== undefined) {
    const interval = time - lastFrameTime;
    frameIntervalEma =
      frameIntervalEma === undefined
        ? interval
        : frameIntervalEma * 0.9 + interval * 0.1;
  }
  lastFrameTime = time;
  if (
    frameIntervalEma === undefined ||
    time - lastRenderStatsUpdate < RENDER_STATS_INTERVAL_MS
  ) {
    return;
  }
  lastRenderStatsUpdate = time;
  const fps = (1000 / frameIntervalEma).toFixed(0);
  const gpu = gpuMs === undefined ? '' : ` · ${gpuMs.toFixed(1)} ms gpu`;
  guiHandles?.setRenderStats(`${fps} fps${gpu}`);
}

function resetRenderStats(): void {
  frameIntervalEma = undefined;
  lastFrameTime = undefined;
  lastRenderStatsUpdate = 0;
  guiHandles?.setRenderStats('—');
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function setStatus(tone: 'busy' | 'error', message: string): void {
  status.dataset.tone = tone;
  status.hidden = false;
  statusMessage.textContent = message;
}

function clearTransientStatus(): void {
  if (status.dataset.tone === 'busy') {
    status.hidden = true;
  }
}

const light = setupLightInput(
  canvas,
  (state) => {
    renderer?.setLight(
      state.lightPosition[0],
      state.lightPosition[1],
      state.lightZ,
    );
    if (!autoOrbitEnabled) {
      requestRender();
    }
  },
  listenerController.signal,
);

/** One request in flight at a time; frames arrive faster than inference runs. */
/** Marks the scene dirty so the next loop tick draws even when locked to the depth rate. */
function requestRender(): void {
  renderRequested = true;
}

function scheduleInference(frame: RenderFrame): void {
  if (inferenceBusy || !depthSource || !renderer) {
    return;
  }
  inferenceBusy = true;
  void runInference(frame).finally(() => {
    inferenceBusy = false;
  });
}

async function runInference(frame: RenderFrame): Promise<void> {
  const estimator = depthSource;
  const activeRenderer = renderer;
  if (!estimator || !activeRenderer) {
    return;
  }
  const generation = inferenceGeneration;
  try {
    const result = await estimator.estimate(
      frame,
      activeRenderer.aspect,
      inferenceResolution,
    );
    inferenceMsEma =
      inferenceMsEma === undefined
        ? result.inferenceMs
        : inferenceMsEma * 0.8 + result.inferenceMs * 0.2;
    guiHandles?.setDepthStats(
      `${inferenceMsEma.toFixed(0)} ms · ${(1000 / inferenceMsEma).toFixed(1)} fps`,
    );
    if (generation !== inferenceGeneration || depthSource !== estimator) {
      // Stale: the source, facing, or model changed while this was in flight.
      return;
    }
    // Depth Anything V2 predicts relative, disparity-style depth — larger means closer,
    // not metric — which is the polarity the original's DepthART pipeline produced and
    // the one `setDepthData` expects, near landing at the high end of normalized z.
    activeRenderer.setDepthData(
      result.data,
      result.width,
      result.height,
      rangeStabilizer.next(result.range),
    );
    requestRender();
  } catch (error) {
    console.error('Depth estimation failed:', error);
  }
}

/**
 * The backend row's tail: how the frame reaches the network. Empty on the worker path,
 * which has no device to share; `gpu input` when the runtime built a second device that
 * the already-built renderer could not move to (see `DepthSource.sharedDevice`).
 */
function describeInputPath(source: DepthSource): string {
  const device = source.sharedDevice;
  if (!device) {
    return '';
  }
  // No renderer yet means this device is the one it is about to be built on.
  return rendererDevice === undefined || rendererDevice === device
    ? ' · shared device'
    : ' · gpu input';
}

/**
 * Loads the pipeline for the requested model size, disposing the previous one. Clears
 * `depthSource` first so `scheduleInference` stops issuing requests during the swap.
 */
async function ensureEstimator(
  size: ModelSize,
  device: DepthDevice,
): Promise<void> {
  if (depthSource?.size === size && depthSource.engine === device) {
    return;
  }
  const previous = depthSource;
  depthSource = undefined;
  guiHandles?.setBackend('loading…');
  if (previous) {
    while (inferenceBusy) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    await previous.dispose().catch(() => {});
  }
  setStatus('busy', `Loading ${size} depth model on ${device}…`);
  const source = await createDepthSource(size, {
    engine: device,
    onProgress: (message) => setStatus('busy', message),
  });
  depthSource = source;
  const backend = source.backend;
  // What actually loaded, not what was requested: for webgpu that includes whether the
  // runtime really initialized a GPU device, and whether the frame reaches it without
  // leaving the GPU.
  guiHandles?.setBackend(
    backend
      ? `${backend.executionProviders} · ${backend.dtype}${
          backend.device === 'webgpu' && !backend.webgpuDeviceActive
            ? ' · no gpu device!'
            : ''
        }${describeInputPath(source)}`
      : device,
  );
  inferenceMsEma = undefined;
  inferencePending = true;
}

/**
 * GUI "engine": reloads the depth model on another ONNX execution provider, live, with
 * no trip through the chooser. On failure — a browser exposing `navigator.ml` that
 * cannot actually build a WebNN graph, say — the dropdown reverts and the model reloads
 * on the previous engine.
 */
async function setEngine(device: DepthDevice): Promise<void> {
  if (device === inferenceDevice) {
    return;
  }
  const previousDevice = inferenceDevice;
  inferenceDevice = device;
  const size = depthSource?.size ?? chooser?.model;
  if (!size) {
    return;
  }
  resetDepthPipeline();
  try {
    await ensureEstimator(size, device);
    clearTransientStatus();
  } catch (error) {
    setStatus('error', `Could not load on ${device}: ${errorMessage(error)}`);
    inferenceDevice = previousDevice;
    guiHandles?.setEngine(previousDevice);
    try {
      await ensureEstimator(size, previousDevice);
      clearTransientStatus();
    } catch {
      // Both engines failed; the error status above already explains the first one.
    }
  }
}

/** Discards all depth state tied to the previous source (and any in-flight inference). */
function resetDepthPipeline(): void {
  inferenceGeneration += 1;
  rangeStabilizer.reset();
  depthSource?.reset();
  renderer?.resetHistory();
}

/**
 * Camera frames publish the latest video frame; they do not render. The loop below runs
 * at the display's refresh rate regardless of the camera's (webcams mostly deliver
 * 30 fps, displays refresh at 60–120 Hz), so light motion stays smooth and the ~1–3 ms
 * relight pass is not throttled to the camera. three's VideoTexture re-uploads only on
 * a new video frame, so the extra renders cost no re-copy.
 */
const camera = new DepthCameraSession(
  video,
  {
    onFrame: (cameraFrame: DepthCameraFrame) => {
      currentFrame = {
        source: cameraFrame.source,
        sourceSize: { width: video.videoWidth, height: video.videoHeight },
        mirror: camera.facingMode === 'user',
        uvTransform: cameraFrame.uvTransform,
        swapAxes: cameraFrame.swapAxes,
      };
      inferencePending = true;
    },
    onError: (error) => {
      stopRenderLoop();
      setStatus('error', `Camera stopped: ${errorMessage(error)}`);
    },
    onEnded: () => {
      stopRenderLoop();
      setStatus('error', 'The camera stream ended.');
    },
  },
  { frameRate: CAMERA_FRAME_RATE, facingMode: 'user' },
);

function stopRenderLoop(): void {
  renderLoopGeneration += 1;
  currentFrame = undefined;
  inferencePending = false;
  renderRequested = true;
  resetRenderStats();
}

/**
 * The single render loop, shared by both sources: one `requestAnimationFrame` per
 * display refresh draws `currentFrame`, and hands it to the depth network whenever the
 * network is idle and the frame is new to it.
 */
function startRenderLoop(): void {
  const generation = ++renderLoopGeneration;
  const step = (time: number): void => {
    if (generation !== renderLoopGeneration) {
      return;
    }
    requestAnimationFrame(step);
    const activeRenderer = renderer;
    const frame = currentFrame;
    if (!activeRenderer || !frame) {
      return;
    }
    light.orbitTick();
    if (inferencePending && !inferenceBusy) {
      inferencePending = false;
      scheduleInference(frame);
    }
    if (lockRenderToDepth && !renderRequested) {
      return;
    }
    renderRequested = false;
    let drew: boolean;
    try {
      drew = activeRenderer.render(frame);
    } catch (error) {
      stopRenderLoop();
      setStatus('error', `Rendering stopped: ${errorMessage(error)}`);
      return;
    }
    clearTransientStatus();
    if (drew) {
      recordRenderedFrame(time, activeRenderer.gpuFrameMs);
    }
  };
  requestAnimationFrame(step);
}

async function setFacing(facing: FacingMode): Promise<void> {
  camera.facingMode = facing === 'front' ? 'user' : 'environment';
  await restartCameraIfActive();
}

async function setCameraResolution(
  resolution: DepthCameraResolution,
): Promise<void> {
  camera.resolution = resolution;
  await restartCameraIfActive();
}

function setInferenceResolution(resolution: InferenceResolution): void {
  inferenceResolution = resolution;
  inferenceMsEma = undefined;
  // The depth field's size changes with the capture size, so drop stale state; the
  // static-image loop also needs one fresh inference at the new size.
  resetDepthPipeline();
  inferencePending = true;
}

async function restartCameraIfActive(): Promise<void> {
  if (chooser?.source !== SourceChoice.CAMERA || !camera.active) {
    return;
  }
  camera.stop();
  try {
    await camera.start();
    resetDepthPipeline();
  } catch (error) {
    setStatus('error', `Could not switch camera: ${errorMessage(error)}`);
  }
}

async function loadDemoImage(): Promise<ImageBitmap> {
  if (!demoImage) {
    const response = await fetch(DEMO_IMAGE_URL, {
      signal: listenerController.signal,
    });
    if (!response.ok) {
      throw new Error(`Demo photo download failed (${response.status}).`);
    }
    demoImage = await createImageBitmap(await response.blob());
  }
  return demoImage;
}

async function startSource(
  source: SourceChoice,
  uploadedImage?: ImageBitmap,
): Promise<void> {
  stopRenderLoop();
  camera.stop();

  if (source === SourceChoice.CAMERA) {
    setStatus('busy', 'Waiting for the camera…');
    await camera.start();
    resetDepthPipeline();
    startRenderLoop();
    return;
  }

  setStatus('busy', 'Preparing the photo…');
  const bitmap =
    source === SourceChoice.UPLOAD && uploadedImage
      ? uploadedImage
      : await loadDemoImage();
  resetDepthPipeline();
  currentFrame = {
    source: bitmap,
    sourceSize: { width: bitmap.width, height: bitmap.height },
    mirror: false,
    uvTransform: IDENTITY_MAT2,
    swapAxes: false,
  };
  inferencePending = true;
  startRenderLoop();
}

function showChooser(errorText?: string): void {
  stopRenderLoop();
  camera.stop();
  inferenceGeneration += 1;
  status.hidden = true;
  chooser?.show(errorText);
}

/**
 * Builds the renderer once, on the depth backend's device when it has one.
 *
 * This is why the model loads before the renderer exists: ONNX Runtime always creates
 * its own `GPUDevice` (it accepts an adapter, never a device), so sharing one means the
 * renderer borrowing the runtime's, which requires the runtime to have initialized
 * first. Nothing is visible in the meantime — the renderer draws nothing until the
 * first depth map arrives.
 */
async function ensureRenderer(device: GPUDevice | undefined): Promise<void> {
  if (renderer) {
    return;
  }
  setStatus('busy', 'Preparing WebGPU…');
  renderer = await DepthRelightingRenderer.create(canvas, device);
  rendererDevice = device;
}

async function start(): Promise<void> {
  if (starting || !chooser) {
    return;
  }
  starting = true;
  chooser.hide();
  try {
    await ensureEstimator(chooser.model, inferenceDevice);
    await ensureRenderer(depthSource?.sharedDevice);
    await startSource(chooser.source, chooser.uploadedImage);
  } catch (error) {
    showChooser(`Could not start: ${errorMessage(error)}`);
  } finally {
    starting = false;
  }
}

async function initialize(): Promise<void> {
  if (!navigator.gpu) {
    setStatus(
      'error',
      'WebGPU is unavailable in this browser. Try Chrome or Edge 121+.',
    );
    return;
  }

  chooser = new SourceChooser(() => void start(), listenerController.signal);
  guiHandles = createGui({
    onSwitchSource: () => showChooser(),
    onSettingsChange: (partial) => {
      renderer?.updateSettings(partial);
      requestRender();
    },
    onFacingChange: (facing) => void setFacing(facing),
    onAutoOrbitChange: (enabled) => {
      autoOrbitEnabled = enabled;
      light.setAutoOrbit(enabled);
      requestRender();
    },
    onLockToDepthChange: (enabled) => {
      lockRenderToDepth = enabled;
      resetRenderStats();
      requestRender();
    },
    onCameraResolutionChange: (resolution) =>
      void setCameraResolution(resolution),
    onDepthResolutionChange: (resolution) => setInferenceResolution(resolution),
    onEngineChange: (device) => void setEngine(device),
  });
  window.addEventListener('resize', requestRender, {
    signal: listenerController.signal,
  });
  showChooser();
}

void initialize();
