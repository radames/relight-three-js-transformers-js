/**
 * Model catalogue + message protocol shared between the main-thread depth-estimator
 * client (depth-estimator.ts) and the inference worker (depth-worker.ts).
 */

export const ModelSize = {
  SMALL: 'small',
  BASE: 'base',
  LARGE: 'large',
} as const;
export type ModelSize = (typeof ModelSize)[keyof typeof ModelSize];

export const MODEL_SIZES: readonly ModelSize[] = [
  ModelSize.SMALL,
  ModelSize.BASE,
  ModelSize.LARGE,
];

export const MODEL_IDS: Record<ModelSize, string> = {
  small: 'onnx-community/depth-anything-v2-small',
  base: 'onnx-community/depth-anything-v2-base',
  large: 'onnx-community/depth-anything-v2-large',
};

/**
 * ONNX Runtime execution provider ("engine") the depth model runs on. transformers.js's
 * `deviceToExecutionProviders` maps 'webgpu' -> 'webgpu', 'wasm' -> 'wasm', and
 * 'webnn-gpu'/'webnn-npu' -> {name:'webnn', deviceType}. There is no silent fallback: an
 * unsupported device throws at session creation.
 */
export const DepthDevice = {
  WEBGPU: 'webgpu',
  WASM: 'wasm',
  WEBNN_GPU: 'webnn-gpu',
  WEBNN_NPU: 'webnn-npu',
} as const;
export type DepthDevice = (typeof DepthDevice)[keyof typeof DepthDevice];

export const DEFAULT_DEVICE: DepthDevice = DepthDevice.WEBGPU;

/**
 * Engines this browser can run, in preference order. The `navigator.gpu` / `navigator.ml`
 * checks are the ones transformers.js's env.js uses, so anything listed here survives
 * `deviceToExecutionProviders`.
 */
export function availableDevices(): DepthDevice[] {
  const devices: DepthDevice[] = [];
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    devices.push(DepthDevice.WEBGPU);
  }
  if (typeof navigator !== 'undefined' && 'ml' in navigator) {
    devices.push(DepthDevice.WEBNN_GPU, DepthDevice.WEBNN_NPU);
  }
  devices.push(DepthDevice.WASM);
  return devices;
}

/** The preferred engine when this browser supports it, else the best one it does. */
export function defaultDevice(): DepthDevice {
  const devices = availableDevices();
  return devices.includes(DEFAULT_DEVICE)
    ? DEFAULT_DEVICE
    : (devices[0] as DepthDevice);
}

export type DepthProgressListener = (message: string) => void;

/** What actually loaded, reported rather than inferred, for the GUI to show. */
export interface BackendInfo {
  readonly device: DepthDevice;
  readonly dtype: string;
  /** ORT execution providers the session was created with, as passed to InferenceSession. */
  readonly executionProviders: string;
  /** True when ORT initialized a WebGPU device for this session. */
  readonly webgpuDeviceActive: boolean;
}

export interface DepthRange {
  readonly min: number;
  readonly max: number;
}

export type DepthWorkerRequest =
  | { type: 'load'; size: ModelSize; device: DepthDevice }
  | { type: 'estimate'; id: number; bitmap: ImageBitmap }
  | { type: 'dispose' };

export type DepthWorkerResponse =
  | { type: 'progress'; message: string }
  | { type: 'ready'; backend: BackendInfo }
  | { type: 'load-error'; message: string }
  | {
      type: 'result';
      id: number;
      data: Float32Array;
      width: number;
      height: number;
      range: DepthRange;
    }
  | { type: 'estimate-error'; id: number; message: string }
  | { type: 'disposed' };
