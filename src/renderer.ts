/**
 * Port of the TypeGPU example's DepthRelightingRenderer, on three.js WebGPU + TSL. The
 * relighting math lives in src/relight/; this file owns GPU resource lifecycle,
 * per-frame uniform writes, and orchestration.
 *
 * Color management: the relight shader does its own gamma, ported unchanged —
 * `pow(cameraColor, GAMMA)` to linear, `pow(tonemap(lit), 1/GAMMA)` back. Texels must
 * therefore arrive still sRGB-encoded, so every source texture keeps `NoColorSpace`.
 * That is not only a node-graph setting: on the WebGPU backend `texture.colorSpace`
 * picks the GPU texture *format*, and an `*-srgb` format decodes in hardware at sample
 * time, stacking a second decode on the shader's own and crushing midtones. (The
 * backend's "Video textures must use a color space with a sRGB transfer function"
 * warning is expected.) The canvas output pass then applies `outputColorSpace`
 * unconditionally, so it and `toneMapping` are both off to avoid re-encoding output the
 * shader has already display-encoded.
 */
import * as THREE from 'three/webgpu';
import type { Mat2 } from './mat2.ts';
import {
  DEFAULT_BULB_SIZE,
  DEFAULT_LIGHT_Z,
  DEFAULT_LIGHT_POSITION,
  LIGHT_Z_MAX as LIGHT_Z_MAX_CONST,
  LIGHT_Z_MIN as LIGHT_Z_MIN_CONST,
  MAX_CANVAS_SIDE,
  MAX_PIXEL_RATIO,
  MOTION_ALPHA,
  RelightMode,
  TEMPORAL_ALPHA,
} from './relight/constants.ts';
import {
  createDepthPrepare,
  type DepthPrepareResources,
} from './relight/depth-prepare.ts';
import { createSurface, type SurfaceResources } from './relight/surface.ts';
import {
  createRelightFragment,
  type RelightFragmentResources,
} from './relight/relight-fragment.ts';
import {
  bulbSizeUniform,
  canvasAspectUniform,
  depthRangeUniform,
  motionAlphaUniform,
  temporalAlphaUniform,
  exposureUniform,
  intensityUniform,
  lightColorUniform,
  lightPositionUniform,
  lightZUniform,
  mirrorUniform,
  modeUniform,
  occlusionUniform,
  reliefUniform,
  resetUniform,
  shadowUniform,
  sourceSizeUniform,
  specularUniform,
  swapAxesUniform,
  uvTransformUniform,
} from './relight/uniforms.ts';

export type ViewMode = 'relit' | 'camera' | 'depth' | 'normals';

export interface RelightSettings {
  intensity: number;
  exposure: number;
  relief: number;
  specular: number;
  shadow: number;
  occlusion: number;
  /** World-space radius of the emissive bulb disc. */
  bulbSize: number;
  lightColor: [number, number, number];
  mode: ViewMode;
}

export const defaultRelightingSettings: RelightSettings = {
  intensity: 3,
  exposure: 0.5,
  relief: 0.85,
  specular: 0.22,
  shadow: 0.7,
  occlusion: 0.55,
  bulbSize: DEFAULT_BULB_SIZE,
  lightColor: [1, 0.72, 0.46],
  mode: 'relit',
};

export const LIGHT_Z_MIN = LIGHT_Z_MIN_CONST;
export const LIGHT_Z_MAX = LIGHT_Z_MAX_CONST;

export interface RenderFrame {
  source: HTMLVideoElement | ImageBitmap;
  sourceSize: { width: number; height: number };
  mirror: boolean;
  uvTransform: Mat2;
  swapAxes: boolean;
}

const MODE_TO_ENUM: Record<ViewMode, number> = {
  relit: RelightMode.RELIT,
  camera: RelightMode.CAMERA,
  depth: RelightMode.DEPTH,
  normals: RelightMode.NORMALS,
};

/** The full set of GPU resources tied to one depth-field size (width x height). */
interface Field {
  width: number;
  height: number;
  depthTexture: THREE.DataTexture;
  depthPrepare: DepthPrepareResources;
  surface: SurfaceResources;
  relight: RelightFragmentResources;
  material: THREE.NodeMaterial;
  quad: THREE.QuadMesh;
}

/**
 * Neutralizes `destroy` on a device three.js did not create.
 *
 * ONNX Runtime destroys its `GPUDevice` as soon as its last WebGPU session is released,
 * which takes the canvas down with it: "The Device was lost" on every frame after a
 * model or engine switch. The renderer needs the device until the page unloads, longer
 * than the runtime intends to keep it. The browser still reclaims it on unload, and a
 * later session just builds a second device and runs there — harmless, for the reasons
 * on `DepthSource.sharedDevice`.
 */
const borrowedDevices = new WeakSet<GPUDevice>();

function borrowDevice(device: GPUDevice): void {
  if (borrowedDevices.has(device)) {
    return;
  }
  borrowedDevices.add(device);
  Object.defineProperty(device, 'destroy', {
    value: () => {},
    writable: true,
    configurable: true,
  });
}

export class DepthRelightingRenderer {
  readonly #canvas: HTMLCanvasElement;
  readonly #renderer: THREE.WebGPURenderer;
  readonly #settings: RelightSettings = { ...defaultRelightingSettings };

  #field: Field | undefined;
  /** Set by a new field size or `resetHistory()`; cleared once the compute passes run. */
  #resetHistory = true;
  /** Set by fresh `setDepthData`; cleared once the compute passes consume it. */
  #depthDirty = false;
  /** `performance.now()` of the previous `setDepthData`, for rate-correcting the blend. */
  #lastDepthTime: number | undefined;

  #videoTextures = new WeakMap<HTMLVideoElement, THREE.VideoTexture>();
  #bitmapTextures = new WeakMap<ImageBitmap, THREE.Texture>();
  #currentCameraTexture: THREE.Texture | undefined;

  #canvasWidth = 0;
  #canvasHeight = 0;

  /** EMA of the render pass's GPU ms, from timestamp queries. Undefined until the
   * first resolve, or forever without `timestamp-query`. */
  #gpuFrameMs: number | undefined;
  /** True while a timestamp resolve is in flight; at most one at a time. */
  #gpuResolving = false;

  private constructor(
    canvas: HTMLCanvasElement,
    renderer: THREE.WebGPURenderer,
  ) {
    this.#canvas = canvas;
    this.#renderer = renderer;
  }

  /**
   * Builds the renderer, optionally on a `device` created elsewhere.
   *
   * Borrowing costs features: three requests every adapter feature for a device it
   * creates, a borrowed one only has what its owner asked for. Nothing here needs an
   * optional feature — the depth field is texel-fetched r32float, so
   * `float32-filterable` never comes up — but `trackTimestamp` self-disables without
   * `timestamp-query`, so the panel's GPU-ms row may read as unavailable.
   */
  static async create(
    canvas: HTMLCanvasElement,
    device?: GPUDevice,
  ): Promise<DepthRelightingRenderer> {
    if (device) {
      borrowDevice(device);
    }
    const renderer = new THREE.WebGPURenderer({
      canvas,
      antialias: false,
      // Per-pass GPU timestamps for the panel's "render" stats row; silently a no-op
      // on devices without the `timestamp-query` feature.
      trackTimestamp: true,
      ...(device ? { device } : {}),
    });
    await renderer.init();
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.outputColorSpace = THREE.NoColorSpace;

    const instance = new DepthRelightingRenderer(canvas, renderer);
    instance.#writeSettingsUniforms();
    lightPositionUniform.value.set(
      DEFAULT_LIGHT_POSITION[0],
      DEFAULT_LIGHT_POSITION[1],
    );
    lightZUniform.value = DEFAULT_LIGHT_Z;
    return instance;
  }

  /**
   * The render target's width/height ratio. The depth-inference capture must crop
   * the source with this exact value so the depth map lines up 1:1 with the render.
   */
  get aspect(): number {
    return this.#canvasHeight > 0 ? this.#canvasWidth / this.#canvasHeight : 1;
  }

  /** Smoothed GPU time of the relight pass per frame in ms, or undefined if unavailable. */
  get gpuFrameMs(): number | undefined {
    return this.#gpuFrameMs;
  }

  updateSettings(partial: Partial<RelightSettings>): void {
    Object.assign(this.#settings, partial);
    this.#writeSettingsUniforms();
  }

  setLight(x: number, y: number, z: number): void {
    lightPositionUniform.value.set(x, y);
    lightZUniform.value = z;
  }

  setDepthData(
    data: Float32Array,
    width: number,
    height: number,
    range: { min: number; max: number },
  ): void {
    this.#ensureField(width, height);
    const field = this.#field;
    if (!field) {
      return;
    }
    const image = field.depthTexture.image as {
      data: Float32Array;
      width: number;
      height: number;
    };
    image.data.set(data);
    field.depthTexture.needsUpdate = true;
    depthRangeUniform.value.set(range.min, range.max);
    this.#writeBlendAlphas();
    this.#depthDirty = true;
  }

  resetHistory(): void {
    this.#resetHistory = true;
    this.#lastDepthTime = undefined;
  }

  /**
   * Draws one frame. Synchronous by design: main.ts calls this once per display
   * refresh, and an async render outliving its rAF tick would overlap the next one.
   *
   * A `VideoTexture` uploads only on a new video frame (three sets `needsUpdate` from
   * `requestVideoFrameCallback`), so rendering at 120 Hz against a 30 fps camera does
   * not re-copy the same image.
   *
   * @returns true if a frame was drawn, false if there is no depth field yet.
   */
  render(frame: RenderFrame): boolean {
    this.#syncCanvasSize();
    this.#updateCameraTexture(frame.source);
    this.#writeFrameUniforms(frame);

    const field = this.#field;
    if (!field) {
      // No depth data has arrived yet; nothing to draw.
      return false;
    }

    if (this.#depthDirty || this.#resetHistory) {
      resetUniform.value = this.#resetHistory;
      this.#renderer.compute([
        field.depthPrepare.computeNode,
        field.surface.computeNode,
      ]);
      this.#depthDirty = false;
      this.#resetHistory = false;
    }

    field.quad.render(this.#renderer);
    this.#sampleGpuTime();
    return true;
  }

  destroy(): void {
    this.#field?.material.dispose();
    this.#field?.depthTexture.dispose();
    this.#field?.surface.surfaceTexture.dispose();
    this.#field = undefined;
    this.#renderer.dispose();
  }

  /**
   * Resolves timestamps, at most one in flight. three's pool returns the duration of
   * the most recent frame's render passes — one relight pass here — so the result is
   * already per-frame. Smoothed because Chrome quantizes timestamps to 100 µs.
   */
  #sampleGpuTime(): void {
    if (this.#gpuResolving) {
      return;
    }
    this.#gpuResolving = true;
    this.#renderer
      .resolveTimestampsAsync(THREE.TimestampQuery.RENDER)
      .then((frameMs) => {
        if (frameMs === undefined || !(frameMs > 0)) {
          return;
        }
        this.#gpuFrameMs =
          this.#gpuFrameMs === undefined
            ? frameMs
            : this.#gpuFrameMs * 0.9 + frameMs * 0.1;
      })
      .catch(() => {
        // Timestamp queries are a diagnostic; never let them break rendering.
      })
      .finally(() => {
        this.#gpuResolving = false;
      });
  }

  /**
   * The original tuned both alphas against a depth field updating every rendered frame
   * (~60 Hz); inference here lands at a few Hz, and the two branches need opposite
   * treatment for that.
   *
   * MOTION_ALPHA (large deltas = the subject moved) is a lag parameter, so it is
   * rescaled to the measured interval: 1 - (1 - alpha)^(dt * 60Hz) preserves the
   * original catch-up time constant. Left alone, the field ghosts several updates
   * behind the video.
   *
   * TEMPORAL_ALPHA (small deltas = inference and sensor noise) is a noise-rejection
   * parameter. Noise is per-sample, not per-time, so it keeps the original per-update
   * constant. Rate-correcting it too would disable smoothing and make static surfaces
   * shimmer in z, most visibly at the bulb's occlusion boundary.
   */
  #writeBlendAlphas(): void {
    const now = performance.now();
    const previous = this.#lastDepthTime;
    this.#lastDepthTime = now;
    const frames =
      previous === undefined
        ? Number.POSITIVE_INFINITY
        : Math.max(1, ((now - previous) / 1000) * 60);
    temporalAlphaUniform.value = TEMPORAL_ALPHA;
    motionAlphaUniform.value = 1 - (1 - MOTION_ALPHA) ** frames;
  }

  #writeSettingsUniforms(): void {
    const settings = this.#settings;
    exposureUniform.value = settings.exposure;
    intensityUniform.value = settings.intensity;
    reliefUniform.value = settings.relief;
    specularUniform.value = settings.specular;
    shadowUniform.value = settings.shadow;
    occlusionUniform.value = settings.occlusion;
    bulbSizeUniform.value = settings.bulbSize;
    lightColorUniform.value.set(
      settings.lightColor[0],
      settings.lightColor[1],
      settings.lightColor[2],
      1,
    );
    modeUniform.value = MODE_TO_ENUM[settings.mode];
  }

  #writeFrameUniforms(frame: RenderFrame): void {
    const [m00, m01, m10, m11] = frame.uvTransform;
    uvTransformUniform.value.set(m00, m01, m10, m11);
    sourceSizeUniform.value.set(
      frame.sourceSize.width,
      frame.sourceSize.height,
    );
    mirrorUniform.value = frame.mirror;
    swapAxesUniform.value = frame.swapAxes;
  }

  #updateCameraTexture(source: HTMLVideoElement | ImageBitmap): void {
    let texture: THREE.Texture;
    if (source instanceof HTMLVideoElement) {
      let videoTexture = this.#videoTextures.get(source);
      if (!videoTexture) {
        videoTexture = new THREE.VideoTexture(source);
        // Texture defaults to flipY = true, which the WebGPU backend honors with a flip
        // blit at upload — but our uv convention (row 0 = top, shared with the bitmap
        // path and the depth-inference capture) expects no flip.
        videoTexture.flipY = false;
        // colorSpace stays NoColorSpace (the default): SRGBColorSpace would allocate
        // an `rgba8unorm-srgb` GPU texture whose hardware decode stacks a second sRGB
        // decode on the shader's own pow(GAMMA) — see the module-level color note.
        this.#videoTextures.set(source, videoTexture);
      }
      texture = videoTexture;
    } else {
      let bitmapTexture = this.#bitmapTextures.get(source);
      if (!bitmapTexture) {
        bitmapTexture = new THREE.Texture(source);
        // Plain Texture wrapping an ImageBitmap auto-flips (flipY defaults to true)
        // unless told otherwise; our uv convention expects no flip.
        bitmapTexture.flipY = false;
        bitmapTexture.needsUpdate = true;
        this.#bitmapTextures.set(source, bitmapTexture);
      }
      texture = bitmapTexture;
    }

    this.#currentCameraTexture = texture;
    if (this.#field) {
      this.#field.relight.cameraTextureNode.value = texture;
    }
  }

  #ensureField(width: number, height: number): void {
    const field = this.#field;
    if (field && field.width === width && field.height === height) {
      return;
    }

    field?.material.dispose();
    field?.depthTexture.dispose();
    field?.surface.surfaceTexture.dispose();

    const depthTexture = new THREE.DataTexture(
      new Float32Array(width * height),
      width,
      height,
      THREE.RedFormat,
      THREE.FloatType,
    );
    depthTexture.needsUpdate = true;

    const depthPrepare = createDepthPrepare(depthTexture, width, height);
    const surface = createSurface(depthPrepare.history, width, height);
    // The camera texture is swapped in immediately after this via #updateCameraTexture
    // on every render() call, so the placeholder passed here is never actually sampled
    // for a real frame.
    const relight = createRelightFragment(
      surface.surfaceTexture,
      depthPrepare.bulbHistory,
      width,
      height,
      this.#currentCameraTexture ?? new THREE.Texture(),
    );
    if (this.#currentCameraTexture) {
      relight.cameraTextureNode.value = this.#currentCameraTexture;
    }

    const material = new THREE.NodeMaterial();
    material.fragmentNode = relight.fragmentNode;
    material.lights = false;
    material.transparent = false;
    material.depthTest = false;
    material.depthWrite = false;

    const quad = new THREE.QuadMesh(material);

    this.#field = {
      width,
      height,
      depthTexture,
      depthPrepare,
      surface,
      relight,
      material,
      quad,
    };
    this.#resetHistory = true;
  }

  #syncCanvasSize(): void {
    const displayWidth = this.#canvas.clientWidth;
    const displayHeight = this.#canvas.clientHeight;
    if (displayWidth <= 0 || displayHeight <= 0) {
      return;
    }
    // Unlike the original (which always rendered a square and let CSS stretch it),
    // the buffer tracks the canvas's real aspect so fullscreen output is undistorted;
    // MAX_CANVAS_SIDE now caps the longest edge.
    const ratio = Math.min(globalThis.devicePixelRatio || 1, MAX_PIXEL_RATIO);
    let width = displayWidth * ratio;
    let height = displayHeight * ratio;
    const longest = Math.max(width, height);
    if (longest > MAX_CANVAS_SIDE) {
      width *= MAX_CANVAS_SIDE / longest;
      height *= MAX_CANVAS_SIDE / longest;
    }
    width = Math.max(1, Math.round(width));
    height = Math.max(1, Math.round(height));
    if (width === this.#canvasWidth && height === this.#canvasHeight) {
      return;
    }
    this.#canvasWidth = width;
    this.#canvasHeight = height;
    canvasAspectUniform.value = width / height;
    // The WebGPU backend must learn about size changes through its own API to
    // reconfigure the canvas context, so `setSize` replaces the original's direct
    // `canvas.width` mutation. `updateStyle = false` leaves CSS sizing to page layout,
    // and `setPixelRatio(1)` keeps it 1:1 since the size is already DPR-scaled.
    this.#renderer.setPixelRatio(1);
    this.#renderer.setSize(width, height, false);
  }
}
