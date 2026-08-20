/**
 * GPU replacement for the depth pipeline's CPU input path.
 *
 * The worker path (capture.ts + depth-worker.ts) draws the frame through an affine
 * transform into a 2D canvas, reads it back with `getImageData`, and lets the image
 * processor rescale, normalize and repack it into an NCHW tensor: three CPU passes over
 * the pixels between a GPU readback and a GPU upload. This module does all of it in one
 * compute dispatch on ONNX Runtime's own `GPUDevice`, writing the tensor straight into
 * a storage buffer the session takes as a `gpu-buffer` input.
 *
 * The UV mapping is a literal WGSL port of `cameraUvAt` (src/relight/camera-uv.ts), so
 * the depth map lines up with the render exactly as the canvas path did. capture.ts has
 * to invert that chain into a forward affine because a 2D canvas can only push pixels
 * forward; a compute shader iterates over destination texels, so it uses the original
 * backward mapping unchanged.
 *
 * Normalization constants come from the loaded image processor rather than hardcoded,
 * so a model shipping different `image_mean`/`image_std` stays correct.
 */
import type { Mat2 } from '../mat2.ts';

/** Matches the `Params` struct in `SHADER`; std140-compatible 16-byte rows. */
const PARAMS_BYTES = 80;
const PARAMS_FLOATS = PARAMS_BYTES / 4;

/** One workgroup covers an 8x8 tile of the destination. */
const WORKGROUP = 8;

/**
 * Must match the session's declared `pixel_values` type: an fp16 export handed float32
 * reads it as garbage. `f16` storage arrays need the `shader-f16` device feature, which
 * is the same condition that selects the fp16 model (see depth-estimator-gpu.ts), so
 * the two always agree.
 */
export type TensorElementType = 'float32' | 'float16';

function buildShader(elementType: TensorElementType): string {
  const f16 = elementType === 'float16';
  return `${f16 ? 'enable f16;\n' : ''}
struct Params {
  sourceSize   : vec2f,
  canvasAspect : f32,
  mirror       : u32,
  swapAxes     : u32,
  outW         : u32,
  outH         : u32,
  _pad         : u32,
  uvTransform  : vec4f,
  mean         : vec4f,
  // .xyz = image_std, .w = the 0..255 scale factor (255 * rescale_factor).
  // Named stddev rather than std, which is a reserved WGSL keyword.
  stddev       : vec4f,
};

@group(0) @binding(0) var srcTex     : texture_2d<f32>;
@group(0) @binding(1) var srcSampler : sampler;
@group(0) @binding(2) var<uniform> params : Params;
@group(0) @binding(3) var<storage, read_write> outBuffer : array<${f16 ? 'f16' : 'f32'}>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP}, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= params.outW || gid.y >= params.outH) {
    return;
  }

  let outSize = vec2f(f32(params.outW), f32(params.outH));
  let uv = (vec2f(f32(gid.x), f32(gid.y)) + 0.5) / outSize;

  // --- port of cameraUvAt (src/relight/camera-uv.ts) ---
  var sourceSize = params.sourceSize;
  if (params.swapAxes != 0u) {
    sourceSize = vec2f(params.sourceSize.y, params.sourceSize.x);
  }

  var framed = uv;
  if (params.mirror != 0u) {
    framed = vec2f(1.0 - uv.x, uv.y);
  }

  let cropHeight = min(sourceSize.y, sourceSize.x / params.canvasAspect);
  let crop = vec2f(cropHeight * params.canvasAspect, cropHeight);
  let sourcePixel = (sourceSize - crop) * 0.5 + framed * crop - 0.5;
  let clamped = clamp(sourcePixel, vec2f(0.0), sourceSize - 1.0);
  let sourceUv = (clamped + 0.5) / sourceSize;

  let centered = sourceUv - 0.5;
  let t = params.uvTransform;
  let sampleUv = vec2f(
    t.x * centered.x + t.y * centered.y,
    t.z * centered.x + t.w * centered.y,
  ) + 0.5;
  // --- end cameraUvAt ---

  let rgba = textureSampleLevel(srcTex, srcSampler, sampleUv, 0.0);

  // The texture is rgba8unorm (never *-srgb), so the sample is the raw stored byte
  // over 255 — the same value getImageData would have produced, just pre-divided.
  // params.stddev.w folds 255 back in together with the processor's rescale_factor.
  let pixel = rgba.rgb * params.stddev.w;
  let normalized = (pixel - params.mean.xyz) / params.stddev.xyz;

  // NCHW: three contiguous H*W planes.
  let plane = params.outW * params.outH;
  let index = gid.y * params.outW + gid.x;
  outBuffer[index] = ${f16 ? 'f16(normalized.r)' : 'normalized.r'};
  outBuffer[plane + index] = ${f16 ? 'f16(normalized.g)' : 'normalized.g'};
  outBuffer[2u * plane + index] = ${f16 ? 'f16(normalized.b)' : 'normalized.b'};
}
`;
}

export interface NormalizationConfig {
  /** Per-channel mean, in the processor's 0..255-scaled units. */
  readonly mean: readonly [number, number, number];
  /** Per-channel standard deviation, same units as `mean`. */
  readonly std: readonly [number, number, number];
  /** The processor's `rescale_factor` (typically 1/255). */
  readonly rescale: number;
}

/**
 * Everything about a frame except the pixels: the framing the shader must reproduce.
 * Split out from `PreprocessFrame` so the dispatch can be driven against a source
 * texture that was filled by something other than a browser image source.
 */
export interface PreprocessGeometry {
  readonly sourceSize: { readonly width: number; readonly height: number };
  readonly mirror: boolean;
  readonly uvTransform: Mat2;
  readonly swapAxes: boolean;
}

export interface PreprocessFrame extends PreprocessGeometry {
  readonly source: HTMLVideoElement | ImageBitmap;
}

/** The NCHW float32 tensor produced by one `run()`, living in GPU memory. */
export interface PreprocessResult {
  readonly buffer: GPUBuffer;
  /** `[1, 3, height, width]`, matching the model's `pixel_values` input. */
  readonly dims: readonly [number, number, number, number];
  /** Element type actually written, to declare on the ONNX Runtime tensor. */
  readonly elementType: TensorElementType;
}

/**
 * Owns the input path's GPU resources: a staging texture, the compute pipeline, and the
 * NCHW output buffer. Reallocated only on a size change, so steady-state inference
 * allocates nothing.
 */
export class GpuPreprocessor {
  readonly #device: GPUDevice;
  readonly #normalization: NormalizationConfig;
  readonly #elementType: TensorElementType;
  readonly #bytesPerElement: number;
  readonly #pipeline: GPUComputePipeline;
  readonly #sampler: GPUSampler;
  readonly #paramsBuffer: GPUBuffer;
  readonly #params = new Float32Array(PARAMS_FLOATS);
  /** The same backing store viewed as u32, for the integer struct members. */
  readonly #paramsU32: Uint32Array;

  #sourceTexture: GPUTexture | undefined;
  #sourceWidth = 0;
  #sourceHeight = 0;

  #outputBuffer: GPUBuffer | undefined;
  #outputWidth = 0;
  #outputHeight = 0;

  #bindGroup: GPUBindGroup | undefined;
  #destroyed = false;

  constructor(
    device: GPUDevice,
    normalization: NormalizationConfig,
    elementType: TensorElementType,
  ) {
    this.#device = device;
    this.#normalization = normalization;
    this.#elementType = elementType;
    this.#bytesPerElement = elementType === 'float16' ? 2 : 4;
    this.#paramsU32 = new Uint32Array(this.#params.buffer);

    const module = device.createShaderModule({
      label: 'depth-preprocess',
      code: buildShader(elementType),
    });
    this.#pipeline = device.createComputePipeline({
      label: 'depth-preprocess',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    this.#sampler = device.createSampler({
      label: 'depth-preprocess',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    this.#paramsBuffer = device.createBuffer({
      label: 'depth-preprocess-params',
      size: PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Uploads `frame`, dispatches the preprocessing pass, and returns the NCHW buffer.
   *
   * The buffer is owned by this instance and reused on the next call, so the consuming
   * inference must be queued before `run` is called again. Only one is ever in flight.
   *
   * @param aspect The render target's aspect ratio. The cover-crop must match the
   *   on-screen framing or the depth map will not line up with the picture.
   */
  run(
    frame: PreprocessFrame,
    aspect: number,
    outWidth: number,
    outHeight: number,
  ): PreprocessResult {
    if (this.#destroyed) {
      throw new Error('The GPU preprocessor was destroyed.');
    }
    const { width, height } = frame.sourceSize;
    if (width <= 0 || height <= 0) {
      throw new Error('The frame source has no dimensions yet.');
    }

    const texture = this.prepare(width, height, outWidth, outHeight);
    this.#device.queue.copyExternalImageToTexture(
      { source: frame.source, flipY: false },
      { texture },
      { width, height },
    );
    return this.dispatch(frame, aspect, outWidth, outHeight);
  }

  /**
   * (Re)allocates the staging texture and the NCHW output buffer for these sizes and
   * returns the texture, so the caller can fill it however it has the pixels.
   */
  prepare(
    sourceWidth: number,
    sourceHeight: number,
    outWidth: number,
    outHeight: number,
  ): GPUTexture {
    const texture = this.#ensureSourceTexture(sourceWidth, sourceHeight);
    this.#ensureOutputBuffer(outWidth, outHeight);
    return texture;
  }

  /**
   * Runs the preprocessing pass over whatever is currently in the staging texture.
   * `prepare` must have been called with matching sizes first.
   */
  dispatch(
    geometry: PreprocessGeometry,
    aspect: number,
    outWidth: number,
    outHeight: number,
  ): PreprocessResult {
    const output = this.#outputBuffer;
    if (!output) {
      throw new Error('`prepare` must run before `dispatch`.');
    }
    this.#writeParams(geometry, aspect, outWidth, outHeight);

    const encoder = this.#device.createCommandEncoder({
      label: 'depth-preprocess',
    });
    const pass = encoder.beginComputePass({ label: 'depth-preprocess' });
    pass.setPipeline(this.#pipeline);
    pass.setBindGroup(0, this.#bindGroup as GPUBindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(outWidth / WORKGROUP),
      Math.ceil(outHeight / WORKGROUP),
    );
    pass.end();
    this.#device.queue.submit([encoder.finish()]);

    return {
      buffer: output,
      dims: [1, 3, outHeight, outWidth],
      elementType: this.#elementType,
    };
  }

  destroy(): void {
    this.#destroyed = true;
    this.#sourceTexture?.destroy();
    this.#sourceTexture = undefined;
    this.#outputBuffer?.destroy();
    this.#outputBuffer = undefined;
    this.#paramsBuffer.destroy();
  }

  #ensureSourceTexture(width: number, height: number): GPUTexture {
    if (
      this.#sourceTexture &&
      this.#sourceWidth === width &&
      this.#sourceHeight === height
    ) {
      return this.#sourceTexture;
    }
    this.#sourceTexture?.destroy();
    // rgba8unorm, never rgba8unorm-srgb: the processor's constants are defined against
    // raw sRGB-encoded bytes, so a hardware decode at sample time would be wrong — the
    // same reason renderer.ts gives for its own textures.
    this.#sourceTexture = this.#device.createTexture({
      label: 'depth-preprocess-source',
      size: { width, height },
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.#sourceWidth = width;
    this.#sourceHeight = height;
    this.#bindGroup = undefined;
    return this.#sourceTexture;
  }

  #ensureOutputBuffer(width: number, height: number): GPUBuffer {
    if (
      this.#outputBuffer &&
      this.#outputWidth === width &&
      this.#outputHeight === height
    ) {
      this.#ensureBindGroup();
      return this.#outputBuffer;
    }
    this.#outputBuffer?.destroy();
    this.#outputBuffer = this.#device.createBuffer({
      label: 'depth-preprocess-nchw',
      size: width * height * 3 * this.#bytesPerElement,
      // STORAGE for the dispatch below; COPY_SRC/COPY_DST because ONNX Runtime
      // registers this as an external input buffer and may copy out of it.
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    this.#outputWidth = width;
    this.#outputHeight = height;
    this.#bindGroup = undefined;
    this.#ensureBindGroup();
    return this.#outputBuffer;
  }

  #ensureBindGroup(): void {
    if (this.#bindGroup || !this.#sourceTexture || !this.#outputBuffer) {
      return;
    }
    this.#bindGroup = this.#device.createBindGroup({
      label: 'depth-preprocess',
      layout: this.#pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.#sourceTexture.createView() },
        { binding: 1, resource: this.#sampler },
        { binding: 2, resource: { buffer: this.#paramsBuffer } },
        { binding: 3, resource: { buffer: this.#outputBuffer } },
      ],
    });
  }

  #writeParams(
    frame: PreprocessGeometry,
    aspect: number,
    outWidth: number,
    outHeight: number,
  ): void {
    const f = this.#params;
    const u = this.#paramsU32;
    const { mean, std, rescale } = this.#normalization;

    f[0] = frame.sourceSize.width;
    f[1] = frame.sourceSize.height;
    f[2] = aspect;
    u[3] = frame.mirror ? 1 : 0;

    u[4] = frame.swapAxes ? 1 : 0;
    u[5] = outWidth;
    u[6] = outHeight;
    u[7] = 0;

    // Row-major Mat2 (see src/mat2.ts): [m00, m01, m10, m11].
    f[8] = frame.uvTransform[0];
    f[9] = frame.uvTransform[1];
    f[10] = frame.uvTransform[2];
    f[11] = frame.uvTransform[3];

    f[12] = mean[0];
    f[13] = mean[1];
    f[14] = mean[2];
    f[15] = 0;

    f[16] = std[0];
    f[17] = std[1];
    f[18] = std[2];
    // Sampled texels are already 0..1; scale back to the processor's units.
    f[19] = 255 * rescale;

    this.#device.queue.writeBuffer(this.#paramsBuffer, 0, f);
  }
}
