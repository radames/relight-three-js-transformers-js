/**
 * The worker depth backend: a `DepthSource` whose inference runs off the main thread.
 *
 * All transformers.js work — model download, inference, CPU pre/post-processing, the
 * percentile scan — happens in the worker so the render loop never blocks. This class
 * captures the frame, then only shuttles `ImageBitmap`s in and depth buffers out, both
 * transferred rather than copied.
 *
 * Two pipeline behaviors this relies on:
 *  - `pipeline('depth-estimation', ...)` accepts an `OffscreenCanvas` directly.
 *  - `predicted_depth` is the raw float tensor (unlike `depth`, the quantized preview),
 *    with dims `[height, width]` matching the input canvas's own size.
 */
import type { RenderFrame } from '../renderer.ts';
import {
  captureBitmapForInference,
  type InferenceResolution,
} from './capture.ts';
import type { DepthEstimate, DepthSource } from './depth-source.ts';
import type {
  BackendInfo,
  DepthDevice,
  DepthProgressListener,
  DepthWorkerRequest,
  DepthWorkerResponse,
  ModelSize,
} from './model.ts';

/** One prediction as the worker sends it; `inferenceMs` is timed by the caller. */
type Prediction = Omit<DepthEstimate, 'inferenceMs'>;

interface PendingEstimate {
  resolve(prediction: Prediction): void;
  reject(error: Error): void;
}

export class DepthEstimator implements DepthSource {
  static async create(
    size: ModelSize,
    engine: DepthDevice,
    onProgress?: DepthProgressListener,
  ): Promise<DepthEstimator> {
    const worker = new Worker(new URL('./depth-worker.ts', import.meta.url), {
      type: 'module',
    });
    const instance = new DepthEstimator(worker, size, engine, onProgress);
    try {
      await instance.#request({ type: 'load', size, device: engine });
    } catch (error) {
      worker.terminate();
      throw error;
    }
    return instance;
  }

  readonly size: ModelSize;
  /** The engine (ONNX execution provider) this estimator was asked to load on. */
  readonly engine: DepthDevice;
  /** The worker's device lives on another thread, so there is nothing to share. */
  readonly sharedDevice = undefined;
  /** What the worker reported after loading — the engine/dtype actually in use. */
  backend: BackendInfo | undefined;
  readonly #worker: Worker;
  readonly #onProgress: DepthProgressListener | undefined;
  readonly #pending = new Map<number, PendingEstimate>();
  #loadWaiter: { resolve(): void; reject(error: Error): void } | undefined;
  #disposeWaiter: (() => void) | undefined;
  #nextId = 0;
  #disposed = false;

  private constructor(
    worker: Worker,
    size: ModelSize,
    engine: DepthDevice,
    onProgress?: DepthProgressListener,
  ) {
    this.#worker = worker;
    this.size = size;
    this.engine = engine;
    this.#onProgress = onProgress;
    worker.addEventListener(
      'message',
      (event: MessageEvent<DepthWorkerResponse>) => this.#onMessage(event.data),
    );
    worker.addEventListener('error', (event) => {
      this.#failEverything(
        new Error(event.message || 'The depth-inference worker crashed.'),
      );
    });
  }

  async estimate(
    frame: RenderFrame,
    aspect: number,
    resolution: InferenceResolution,
  ): Promise<DepthEstimate> {
    const start = performance.now();
    const bitmap = await captureBitmapForInference(frame, aspect, resolution);
    const prediction = await this.#predict(bitmap);
    return { ...prediction, inferenceMs: performance.now() - start };
  }

  reset(): void {
    // The worker owns its aligner and only resets it on model load, so a source change
    // costs one frame of stale alignment reference. Reaching in would need a new worker
    // message for a difference that decays within that frame.
  }

  /** Releases the worker and its ONNX sessions, on a model-size switch. */
  async dispose(): Promise<void> {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    await new Promise<void>((resolve) => {
      this.#disposeWaiter = resolve;
      this.#post({ type: 'dispose' });
      // Don't hang forever if the worker is wedged; terminate reclaims it either way.
      setTimeout(resolve, 3000);
    });
    this.#worker.terminate();
    this.#failEverything(new Error('The depth estimator was disposed.'));
  }

  #predict(bitmap: ImageBitmap): Promise<Prediction> {
    if (this.#disposed) {
      bitmap.close();
      return Promise.reject(new Error('The depth estimator was disposed.'));
    }
    const id = this.#nextId++;
    return new Promise<Prediction>((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#post({ type: 'estimate', id, bitmap }, [bitmap]);
    });
  }

  #post(message: DepthWorkerRequest, transfer?: Transferable[]): void {
    this.#worker.postMessage(message, transfer ?? []);
  }

  #request(message: DepthWorkerRequest & { type: 'load' }): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      this.#loadWaiter = { resolve, reject };
      this.#post(message);
    });
  }

  #onMessage(message: DepthWorkerResponse): void {
    switch (message.type) {
      case 'progress':
        this.#onProgress?.(message.message);
        break;
      case 'ready':
        this.backend = message.backend;
        this.#loadWaiter?.resolve();
        this.#loadWaiter = undefined;
        break;
      case 'load-error':
        this.#loadWaiter?.reject(new Error(message.message));
        this.#loadWaiter = undefined;
        break;
      case 'result': {
        const pending = this.#pending.get(message.id);
        this.#pending.delete(message.id);
        pending?.resolve({
          data: message.data,
          width: message.width,
          height: message.height,
          range: message.range,
        });
        break;
      }
      case 'estimate-error': {
        const pending = this.#pending.get(message.id);
        this.#pending.delete(message.id);
        pending?.reject(new Error(message.message));
        break;
      }
      case 'disposed':
        this.#disposeWaiter?.();
        this.#disposeWaiter = undefined;
        break;
    }
  }

  #failEverything(error: Error): void {
    this.#loadWaiter?.reject(error);
    this.#loadWaiter = undefined;
    for (const pending of this.#pending.values()) {
      pending.reject(error);
    }
    this.#pending.clear();
  }
}
