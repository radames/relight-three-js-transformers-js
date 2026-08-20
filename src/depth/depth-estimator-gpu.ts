/**
 * Main-thread depth estimator that keeps the network's *input* on the GPU.
 *
 * It runs on the main thread despite the worker path existing to avoid exactly that: a
 * WebGPU device cannot cross a thread boundary, so a `GPUBuffer` produced in a worker
 * means nothing to a renderer on the main thread. The CPU work that justified the
 * worker — the 2D-canvas draw, the `getImageData` readback, the processor's
 * rescale/normalize/NCHW-pack — is what this path deletes, leaving a compute dispatch
 * and an async `session.run`.
 *
 * Device ownership runs backwards from the obvious design. ONNX Runtime never accepts
 * an externally created device (`initEp` takes an *adapter*; the device is always made
 * inside the runtime), while three.js does accept one. So the runtime creates it and
 * the renderer borrows it, which is why `create()` must be awaited before the renderer
 * is constructed. That costs nothing visually — the renderer draws nothing until the
 * first depth map arrives.
 *
 * Scope: the input is on-GPU, but the prediction is still downloaded and the
 * percentile/alignment scan still runs on the CPU, as on the worker path.
 */
import {
  env,
  pipeline,
  Tensor,
  type DepthEstimationPipeline,
  type ProgressInfo,
} from '@huggingface/transformers';
// Imported directly, pinned to the version transformers.js depends on so the bundler
// resolves one shared instance, purely for `Tensor.fromGpuBuffer` — which the
// transformers.js tensor wrapper does not expose.
import { Tensor as OrtTensor } from 'onnxruntime-web/webgpu';
import type { RenderFrame } from '../renderer.ts';
import { captureSize, type InferenceResolution } from './capture.ts';
import type { DepthEstimate, DepthSource } from './depth-source.ts';
import {
  DepthDevice,
  MODEL_IDS,
  type BackendInfo,
  type DepthProgressListener,
  type ModelSize,
} from './model.ts';
import {
  GpuPreprocessor,
  type NormalizationConfig,
  type TensorElementType,
} from './gpu-preprocess.ts';
import { DepthAligner, percentileRange } from './percentile.ts';

/**
 * The pipeline surface used when bypassing `DepthEstimationPipeline._call`. Both fields
 * are public `Callable` instances whose call signatures the published typings omit,
 * hence this cast target. `model.sessions` is the raw ONNX Runtime session map, read
 * only for the declared input type.
 */
interface DirectDepthPipeline {
  model: ((inputs: Record<string, unknown>) => Promise<{
    predicted_depth: { dims: number[]; data: Float32Array | number[] };
  }>) & {
    sessions: Record<
      string,
      {
        inputNames: readonly string[];
        inputMetadata: readonly {
          name: string;
          isTensor: boolean;
          type?: string;
        }[];
      }
    >;
  };
  processor: {
    image_processor: {
      image_mean?: number[] | number;
      image_std?: number[] | number;
      rescale_factor?: number;
      do_resize: boolean;
    };
  };
}

function describeProgress(info: ProgressInfo): string | undefined {
  switch (info.status) {
    case 'initiate':
      return `Preparing ${info.file}…`;
    case 'download':
      return `Downloading ${info.file}…`;
    case 'progress':
      return `Downloading model ${Math.round(info.progress)}%`;
    case 'ready':
      return 'Depth model ready';
    default:
      return undefined;
  }
}

async function supportsShaderF16(): Promise<boolean> {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    return Boolean(adapter?.features.has('shader-f16'));
  } catch {
    return false;
  }
}

/** Broadcasts a scalar `image_mean`/`image_std` to the three colour channels. */
function toTriple(
  value: number[] | number | undefined,
  fallback: number,
): [number, number, number] {
  if (typeof value === 'number') {
    return [value, value, value];
  }
  if (Array.isArray(value) && value.length >= 3) {
    return [value[0] as number, value[1] as number, value[2] as number];
  }
  return [fallback, fallback, fallback];
}

/**
 * Reads the runtime's `GPUDevice`. onnxruntime-web only populates `env.webgpu.device`
 * from inside WebGPU EP initialization, so this is undefined before the first load and
 * afterwards is proof the EP is live rather than having quietly landed on wasm.
 */
function runtimeGpuDevice(): GPUDevice | undefined {
  const onnx = env.backends.onnx as unknown as
    { webgpu?: { device?: GPUDevice } } | undefined;
  return onnx?.webgpu?.device;
}

export class GpuDepthEstimator implements DepthSource {
  /**
   * Loads the model on the WebGPU EP and builds the preprocessing pass against the
   * device the runtime created. Throws if the WebGPU EP did not come up, so callers can
   * fall back to the worker path.
   */
  static async create(
    size: ModelSize,
    onProgress?: DepthProgressListener,
  ): Promise<GpuDepthEstimator> {
    const dtype = (await supportsShaderF16()) ? 'fp16' : 'fp32';
    const loaded = await pipeline('depth-estimation', MODEL_IDS[size], {
      device: 'webgpu',
      dtype,
      progress_callback: (info: ProgressInfo) => {
        const message = describeProgress(info);
        if (message) {
          onProgress?.(message);
        }
      },
    });

    const device = runtimeGpuDevice();
    if (!device) {
      throw new Error(
        'ONNX Runtime did not initialize a WebGPU device; the session likely fell back to another execution provider.',
      );
    }

    const direct = loaded as unknown as DirectDepthPipeline;
    // The capture size is already a multiple of Depth Anything's 14px patch size, and
    // the processor's DPT keep_aspect_ratio logic would rescale it back toward the
    // 518px config target, undoing the chosen inference resolution. This path never
    // calls the processor, but the flag is cleared so the two paths cannot disagree
    // about what resolution the network saw.
    direct.processor.image_processor.do_resize = false;

    const imageProcessor = direct.processor.image_processor;
    const normalization: NormalizationConfig = {
      mean: toTriple(imageProcessor.image_mean, 0.5),
      std: toTriple(imageProcessor.image_std, 0.5),
      rescale: imageProcessor.rescale_factor ?? 1 / 255,
    };

    const inputElementType = resolveInputElementType(direct);
    if (inputElementType === 'float16' && !device.features.has('shader-f16')) {
      throw new Error(
        'The model expects a float16 input but the device lacks `shader-f16`.',
      );
    }

    const preprocessor = new GpuPreprocessor(
      device,
      normalization,
      inputElementType,
    );
    return new GpuDepthEstimator(size, loaded, direct, device, preprocessor, {
      device: DepthDevice.WEBGPU,
      dtype,
      executionProviders: 'webgpu',
      // True by construction: `create` throws above without a runtime device.
      webgpuDeviceActive: true,
    });
  }

  readonly size: ModelSize;
  readonly engine = DepthDevice.WEBGPU;
  readonly sharedDevice: GPUDevice;
  readonly backend: BackendInfo;
  readonly #pipeline: DepthEstimationPipeline;
  readonly #direct: DirectDepthPipeline;
  readonly #preprocessor: GpuPreprocessor;
  readonly #aligner = new DepthAligner();
  #disposed = false;

  private constructor(
    size: ModelSize,
    loaded: DepthEstimationPipeline,
    direct: DirectDepthPipeline,
    device: GPUDevice,
    preprocessor: GpuPreprocessor,
    backend: BackendInfo,
  ) {
    this.size = size;
    this.#pipeline = loaded;
    this.#direct = direct;
    this.sharedDevice = device;
    this.#preprocessor = preprocessor;
    this.backend = backend;
  }

  /**
   * Runs one estimate against `frame`, cropped to `aspect` at `resolution`.
   *
   * A compute dispatch writes the input tensor into a storage buffer, handed to the
   * session as a `gpu-buffer` tensor. The transformers.js `Tensor` wrapper takes a
   * runtime tensor verbatim and `validateInputs` never touches `.data`, so it survives
   * the normal call path — leaving the runtime to decode an fp16 output back to
   * float32, as on the worker path.
   */
  async estimate(
    frame: RenderFrame,
    aspect: number,
    resolution: InferenceResolution,
  ): Promise<DepthEstimate> {
    if (this.#disposed) {
      throw new Error('The depth estimator was disposed.');
    }

    const start = performance.now();
    const { width: outWidth, height: outHeight } = captureSize(
      aspect,
      resolution,
    );
    const { buffer, dims, elementType } = this.#preprocessor.run(
      frame,
      aspect,
      outWidth,
      outHeight,
    );

    const ortTensor = OrtTensor.fromGpuBuffer(buffer, {
      dataType: elementType,
      dims,
    });
    const pixelValues = new Tensor(ortTensor);

    const { predicted_depth } = await this.#direct.model({
      pixel_values: pixelValues,
    });

    // Dims are [1, height, width]. Depth Anything's head outputs at input resolution
    // for patch-aligned inputs, which the caller guarantees.
    const shape = predicted_depth.dims;
    const height = shape[shape.length - 2] ?? 0;
    const width = shape[shape.length - 1] ?? 0;
    const raw = predicted_depth.data;
    const data =
      raw instanceof Float32Array
        ? raw
        : Float32Array.from(raw as ArrayLike<number>);

    this.#aligner.align(data);
    const range = percentileRange(data);
    // Timed to here, not to `model()`: the alignment and percentile scans are part of
    // the frame-to-prediction span the panel reports as the depth-update period.
    return {
      data,
      width,
      height,
      range,
      inferenceMs: performance.now() - start,
    };
  }

  /** Clears the frame-to-frame alignment reference (on source or facing change). */
  reset(): void {
    this.#aligner.reset();
  }

  async dispose(): Promise<void> {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    this.#preprocessor.destroy();
    await this.#pipeline.dispose().catch(() => {});
  }
}

/**
 * The declared element type of the session's `pixel_values` input. The preprocessing
 * pass matches whatever the graph declares rather than assuming, since float32 written
 * into an fp16 input reads as garbage.
 */
function resolveInputElementType(
  direct: DirectDepthPipeline,
): TensorElementType {
  const session = direct.model.sessions?.['model'];
  const metadata = session?.inputMetadata?.find(
    (entry) => entry.name === 'pixel_values' && entry.isTensor,
  );
  return metadata?.type === 'float16' ? 'float16' : 'float32';
}
