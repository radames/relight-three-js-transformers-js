/**
 * One interface over the two depth backends, so the render loop doesn't care which runs.
 * The engine picks the backend; there is no separate knob.
 *
 *  - **shared device** (`GpuDepthEstimator`) — the `webgpu` engine. Runs on the main
 *    thread so it can share ONNX Runtime's `GPUDevice` with the renderer (`GPUBuffer`s
 *    are not transferable): the frame goes from the video element into a compute pass
 *    that writes the NCHW tensor into a storage buffer the session reads in place.
 *
 *  - **worker** (`DepthEstimator`) — every other engine, and the fallback whenever the
 *    shared-device path cannot come up. A 2D-canvas capture goes to a Web Worker running
 *    the full transformers.js pipeline, which transfers a `Float32Array` back: inference
 *    never touches the main thread, but every frame makes a round trip through system
 *    memory before reaching the network.
 */
import type { RenderFrame } from '../renderer.ts';
import type { InferenceResolution } from './capture.ts';
import { DepthEstimator } from './depth-estimator.ts';
import { GpuDepthEstimator } from './depth-estimator-gpu.ts';
import {
  DepthDevice,
  type BackendInfo,
  type DepthProgressListener,
  type DepthRange,
  type ModelSize,
} from './model.ts';

export interface DepthEstimate {
  readonly data: Float32Array;
  readonly width: number;
  readonly height: number;
  /** 2%–98% percentile range of `data`. */
  readonly range: DepthRange;
  /**
   * Wall-clock ms from having a frame to holding its prediction, capture included,
   * measured on the main thread. Inference is single-flight off the render loop, so this
   * span is also the real depth-update period and `1000 / inferenceMs` an honest frame
   * rate.
   */
  readonly inferenceMs: number;
}

export interface DepthSource {
  readonly size: ModelSize;
  /** The ONNX execution provider ("engine") the network is running on. */
  readonly engine: DepthDevice;
  /** What actually loaded, for the panel's backend row; undefined until reported. */
  readonly backend: BackendInfo | undefined;
  /**
   * The `GPUDevice` the network runs on, for the renderer to be built on. Undefined on
   * the worker path, whose device lives on another thread. Named apart from `engine`
   * because both are colloquially "the device".
   *
   * The renderer only borrows it, best-effort. The runtime releases its device with its
   * last WebGPU session, so reloading the model — a size change, an engine round trip —
   * hands back a new one an already-built renderer cannot move to. That is survivable
   * because nothing here reads the renderer's device: `GpuPreprocessor` copies the frame
   * into its own texture on whichever device the runtime is on, and the prediction
   * returns through system memory into a `DataTexture` on the renderer's. Only reading
   * the model's output buffer straight from a TSL node would need a single device, so a
   * second one costs the panel's label, not the optimization.
   */
  readonly sharedDevice: GPUDevice | undefined;
  estimate(
    frame: RenderFrame,
    aspect: number,
    resolution: InferenceResolution,
  ): Promise<DepthEstimate>;
  /** Drops frame-to-frame state after a source or facing change. */
  reset(): void;
  dispose(): Promise<void>;
}

/**
 * Loads the requested model on the backend its engine implies: `webgpu` on the
 * shared-device path, every other engine — and anything the shared-device path cannot
 * start — in the worker.
 */
export async function createDepthSource(
  size: ModelSize,
  options: {
    engine?: DepthDevice;
    onProgress?: DepthProgressListener;
  } = {},
): Promise<DepthSource> {
  const { onProgress } = options;
  const engine = options.engine ?? DepthDevice.WEBGPU;
  // The shared-device path is a WebGPU arrangement by definition — it exists to hand the
  // runtime's own `GPUDevice` to the renderer — so any other engine goes to the worker.
  if (engine === DepthDevice.WEBGPU) {
    try {
      return await GpuDepthEstimator.create(size, onProgress);
    } catch (error) {
      console.warn(
        'Shared-device depth path unavailable; falling back to the worker.',
        error,
      );
    }
  }
  return DepthEstimator.create(size, engine, onProgress);
}
