/**
 * Depth-inference worker. Runs the `depth-estimation` pipeline and the percentile scan
 * over its output off the main thread, so CPU-side pre/post-processing cannot stall the
 * render loop. WebGPU is available in dedicated workers in every browser that can run
 * this app's renderer.
 *
 * The caller picks the execution provider (`load.device`, the GUI's "engine" dropdown);
 * the resolved engine and dtype come back in the `ready` message so the UI reports what
 * actually loaded.
 *
 * Protocol: see model.ts. The client guarantees ordering — `estimate` only after
 * `ready`, `dispose` only once no estimate is in flight.
 */
import {
  env,
  pipeline,
  RawImage,
  type DepthEstimationPipeline,
  type ProgressInfo,
} from '@huggingface/transformers';
import {
  DepthDevice,
  MODEL_IDS,
  type BackendInfo,
  type DepthWorkerRequest,
  type DepthWorkerResponse,
} from './model.ts';
import { DepthAligner, percentileRange } from './percentile.ts';

// The DOM lib types `self` as `Window`; narrow it to the worker-side surface we use.
const scope = self as unknown as {
  postMessage(message: DepthWorkerResponse, transfer?: Transferable[]): void;
  addEventListener(
    type: 'message',
    listener: (event: MessageEvent<DepthWorkerRequest>) => void,
  ): void;
};

let estimator: DepthEstimationPipeline | undefined;
/** Cancels the model's global scale/offset flicker before results leave the worker. */
const aligner = new DepthAligner();

function describeProgress(info: ProgressInfo): string | undefined {
  switch (info.status) {
    case 'initiate':
      return `Preparing ${info.file}…`;
    case 'download':
      return `Downloading ${info.file}…`;
    case 'progress':
      return `Downloading model ${Math.round(info.progress)}%`;
    case 'done':
      return undefined;
    case 'ready':
      return 'Depth model ready';
    default:
      return undefined;
  }
}

type Dtype = 'fp16' | 'fp32' | 'q8';

// q4f16 was benchmarked here and measured ~3.5x slower than fp16 on the WebGPU EP
// (158ms vs 44ms per frame at 392px on an M-series Mac): its int4 matmul path does not
// pay off in this build. Keep fp16.
//
// Per engine:
//  - webgpu: fp16 when the adapter has shader-f16 (transformers.js throws if it doesn't
//    and fp16 was asked for — see models/session.js), else fp32.
//  - wasm:   the CPU EP has no fp16 kernels; q8 is the only variant that runs at a
//    watchable rate there (fp32 is several seconds per frame for these models).
//  - webnn:  fp32; the WebNN EP wants a float graph, and dynamic shapes already cost it.
async function resolveDtype(device: DepthDevice): Promise<Dtype> {
  if (device === DepthDevice.WASM) {
    return 'q8';
  }
  if (device !== DepthDevice.WEBGPU) {
    return 'fp32';
  }
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    return adapter?.features.has('shader-f16') ? 'fp16' : 'fp32';
  } catch {
    return 'fp32';
  }
}

/**
 * Mirrors transformers.js's own DEVICE_TO_EXECUTION_PROVIDER_MAPPING so the reported
 * string is the EP list actually handed to `InferenceSession.create` for this device.
 */
function executionProvidersFor(device: DepthDevice): string {
  switch (device) {
    case DepthDevice.WEBGPU:
      return 'webgpu';
    case DepthDevice.WASM:
      return 'wasm';
    case DepthDevice.WEBNN_GPU:
      return 'webnn(gpu)';
    case DepthDevice.WEBNN_NPU:
      return 'webnn(npu)';
  }
}

/**
 * ORT populates `env.webgpu.device` from inside `webgpuInit`, which runs only for a
 * session created with the webgpu EP, and transformers.js re-exports that env by
 * reference. A device here is proof the EP initialized rather than the session quietly
 * landing on wasm.
 */
function webgpuDeviceActive(): boolean {
  const onnx = env.backends.onnx as unknown as
    { webgpu?: { device?: unknown } } | undefined;
  return Boolean(onnx?.webgpu?.device);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function load(
  size: keyof typeof MODEL_IDS,
  device: DepthDevice,
): Promise<BackendInfo> {
  const dtype = await resolveDtype(device);
  // Assigned via a local so the `estimator` variable's `| undefined` type doesn't
  // contextually re-infer pipeline()'s generic across every task (TS2590).
  const loaded = await pipeline('depth-estimation', MODEL_IDS[size], {
    device,
    dtype,
    progress_callback: (info: ProgressInfo) => {
      const message = describeProgress(info);
      if (message) {
        scope.postMessage({ type: 'progress', message });
      }
    },
  });
  // Captures are already sized to multiples of 14 (Depth Anything's patch size), so the
  // processor's resize is disabled. Left on, its DPT keep_aspect_ratio logic rescales
  // every input back toward the 518px config target, undoing the client's choice of
  // inference resolution and the speed it buys.
  (
    loaded.processor.image_processor as unknown as { do_resize: boolean }
  ).do_resize = false;
  estimator = loaded;
  aligner.reset();
  return {
    device,
    dtype,
    executionProviders: executionProvidersFor(device),
    webgpuDeviceActive: webgpuDeviceActive(),
  };
}

// Reused across estimates; reallocating a canvas + context per frame is wasteful.
let inputCanvas: OffscreenCanvas | undefined;

function drawToInputCanvas(bitmap: ImageBitmap): OffscreenCanvas {
  if (
    !inputCanvas ||
    inputCanvas.width !== bitmap.width ||
    inputCanvas.height !== bitmap.height
  ) {
    inputCanvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  }
  const ctx = inputCanvas.getContext('2d');
  if (!ctx) {
    throw new Error('Could not acquire a 2D context in the depth worker.');
  }
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return inputCanvas;
}

/**
 * The output tensor's `data` may be a view into a larger pooled buffer; transferring
 * requires exclusive ownership of the whole ArrayBuffer, so copy in that case.
 */
function transferable(data: Float32Array): Float32Array {
  const ownsBuffer =
    data.byteOffset === 0 && data.byteLength === data.buffer.byteLength;
  return ownsBuffer ? data : data.slice();
}

/**
 * The pipeline surface `estimate` uses when bypassing `DepthEstimationPipeline._call`.
 * Both fields are public `Callable` instances whose call signatures the published
 * typings omit, hence this cast target.
 */
interface DirectDepthPipeline {
  processor(images: RawImage[]): Promise<Record<string, unknown>>;
  model(inputs: Record<string, unknown>): Promise<{
    predicted_depth: { dims: number[]; data: Float32Array | number[] };
  }>;
}

async function estimate(id: number, bitmap: ImageBitmap): Promise<void> {
  if (!estimator) {
    throw new Error('Depth model is not loaded.');
  }
  const canvas = drawToInputCanvas(bitmap);

  // Processor + model directly rather than `estimator(canvas)`. `_call` would also run
  // interpolate_4d — a full wasm session dispatch — to resize the prediction to the
  // input size, a no-op here since the head already outputs at input resolution for the
  // patch-aligned inputs capture.ts guarantees, and then spend ~7 more full-array
  // passes building a quantized preview image this app discards.
  const direct = estimator as unknown as DirectDepthPipeline;
  const image = await RawImage.read(canvas);
  const inputs = await direct.processor([image]);
  const { predicted_depth } = await direct.model(inputs);

  // Dims are [1, height, width] at the input canvas's resolution; already Float32Array.
  const tensor = predicted_depth;
  const dims = tensor.dims;
  const height = dims[dims.length - 2] ?? 0;
  const width = dims[dims.length - 1] ?? 0;
  const raw = tensor.data;
  const data = transferable(
    raw instanceof Float32Array
      ? raw
      : Float32Array.from(raw as ArrayLike<number>),
  );
  aligner.align(data);
  const range = percentileRange(data);
  scope.postMessage({ type: 'result', id, data, width, height, range }, [
    data.buffer,
  ]);
}

scope.addEventListener('message', (event) => {
  const message = event.data;
  switch (message.type) {
    case 'load':
      void load(message.size, message.device)
        .then((backend) => scope.postMessage({ type: 'ready', backend }))
        .catch((error: unknown) =>
          scope.postMessage({
            type: 'load-error',
            message: errorMessage(error),
          }),
        );
      break;
    case 'estimate':
      void estimate(message.id, message.bitmap).catch((error: unknown) =>
        scope.postMessage({
          type: 'estimate-error',
          id: message.id,
          message: errorMessage(error),
        }),
      );
      break;
    case 'dispose':
      void (estimator?.dispose() ?? Promise.resolve())
        .catch(() => {})
        .then(() => scope.postMessage({ type: 'disposed' }));
      break;
  }
});
