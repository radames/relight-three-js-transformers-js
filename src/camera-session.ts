import { IDENTITY_MAT2, type Mat2 } from './mat2.ts';

export interface DepthCameraFrame {
  readonly source: HTMLVideoElement;
  readonly uvTransform: Mat2;
  readonly swapAxes: boolean;
}

interface DepthCameraSessionCallbacks {
  readonly onFrame: (frame: DepthCameraFrame) => void;
  readonly onError?: (error: unknown) => void;
  readonly onEnded?: () => void;
}

type DepthCameraFacing = 'user' | 'environment';

export interface DepthCameraResolution {
  readonly width: number;
  readonly height: number;
}

interface DepthCameraSessionOptions {
  readonly frameRate?: number;
  readonly facingMode?: DepthCameraFacing;
  readonly resolution?: DepthCameraResolution;
}

const DEFAULT_RESOLUTION: DepthCameraResolution = { width: 1280, height: 720 };

const IS_IOS = /iPad|iPhone|iPod/.test(navigator.userAgent);

type FrameTransform = Pick<DepthCameraFrame, 'uvTransform' | 'swapAxes'>;

const UPRIGHT_TRANSFORM: FrameTransform = {
  uvTransform: IDENTITY_MAT2,
  swapAxes: false,
};

/**
 * Only iOS hands the capture over in the device's own orientation.
 *
 * The original wrote these as `d.mat2x2f(0, -1, 1, 0)` and friends. TypeGPU's `mat2x2f`
 * constructor is column-major while src/mat2.ts's `Mat2` is row-major, so the portrait
 * tuples here are the transpose of the original's literals — copied verbatim they would
 * apply the inverse rotation. The landscape tuple is symmetric, so it is unchanged.
 */
const IOS_TRANSFORMS: Partial<Record<OrientationType, FrameTransform>> = {
  'portrait-primary': { uvTransform: [0, 1, -1, 0], swapAxes: true },
  'portrait-secondary': { uvTransform: [0, -1, 1, 0], swapAxes: true },
  'landscape-primary': { uvTransform: [-1, 0, 0, -1], swapAxes: false },
};

function frameTransform(): FrameTransform {
  const orientation = screen.orientation?.type;
  if (!IS_IOS || !orientation) {
    return UPRIGHT_TRANSFORM;
  }
  return IOS_TRANSFORMS[orientation] ?? UPRIGHT_TRANSFORM;
}

export class DepthCameraSession {
  readonly #video: HTMLVideoElement;
  readonly #callbacks: DepthCameraSessionCallbacks;
  readonly #frameRate: number | undefined;
  #facingMode: DepthCameraFacing;
  #resolution: DepthCameraResolution;
  #generation = 0;
  #starting = false;
  #stream: MediaStream | undefined;
  #cancelFrame: (() => void) | undefined;
  #disposed = false;

  constructor(
    video: HTMLVideoElement,
    callbacks: DepthCameraSessionCallbacks,
    options: DepthCameraSessionOptions = {},
  ) {
    this.#video = video;
    this.#callbacks = callbacks;
    this.#frameRate = options.frameRate;
    this.#facingMode = options.facingMode ?? 'environment';
    this.#resolution = options.resolution ?? DEFAULT_RESOLUTION;
  }

  get facingMode(): DepthCameraFacing {
    return this.#facingMode;
  }

  /** Takes effect on the next start, so a running session must be restarted */
  set facingMode(value: DepthCameraFacing) {
    this.#facingMode = value;
  }

  get resolution(): DepthCameraResolution {
    return this.#resolution;
  }

  /** Takes effect on the next start, so a running session must be restarted */
  set resolution(value: DepthCameraResolution) {
    this.#resolution = value;
  }

  get active(): boolean {
    return this.#starting || this.#stream !== undefined;
  }

  async start(): Promise<void> {
    if (this.active) {
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        'Camera capture is unavailable in this browser or page context.',
      );
    }

    const generation = ++this.#generation;
    this.#starting = true;
    let nextStream: MediaStream | undefined;
    try {
      const constraints: MediaTrackConstraints = {
        facingMode: { ideal: this.#facingMode },
        width: { ideal: this.#resolution.width },
        height: { ideal: this.#resolution.height },
      };
      if (this.#frameRate !== undefined) {
        constraints.frameRate = {
          ideal: this.#frameRate,
          max: this.#frameRate,
        };
      }
      nextStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: constraints,
      });
      if (!this.#isCurrent(generation)) {
        this.#stopTracks(nextStream);
        return;
      }

      this.#stream = nextStream;
      nextStream = undefined;
      this.#video.srcObject = this.#stream;
      await this.#video.play();
      if (!this.#isCurrent(generation) || !this.#stream) {
        return;
      }

      this.#starting = false;
      this.#video.hidden = false;
      for (const track of this.#stream.getVideoTracks()) {
        track.addEventListener(
          'ended',
          () => {
            if (this.#isCurrent(generation) && this.#stream) {
              this.#release();
              this.#callbacks.onEnded?.();
            }
          },
          { once: true },
        );
      }
      this.#scheduleFrame(generation);
    } catch (error) {
      this.#stopTracks(nextStream);
      if (this.#isCurrent(generation)) {
        this.#release();
      }
      throw error;
    } finally {
      if (this.#isCurrent(generation) && this.#starting) {
        this.#starting = false;
      }
    }
  }

  stop(): void {
    this.#release();
  }

  destroy(): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    this.stop();
  }

  #scheduleFrame(generation: number): void {
    if (!this.#isRunning(generation) || this.#cancelFrame) {
      return;
    }

    const run = (sourceWidth: number, sourceHeight: number): void => {
      this.#cancelFrame = undefined;
      if (
        !this.#isRunning(generation) ||
        sourceWidth <= 0 ||
        sourceHeight <= 0
      ) {
        return;
      }
      try {
        this.#callbacks.onFrame({
          source: this.#video,
          ...frameTransform(),
        });
      } catch (error) {
        this.#release();
        this.#callbacks.onError?.(error);
        return;
      }
      this.#scheduleFrame(generation);
    };

    if (this.#video.requestVideoFrameCallback) {
      const handle = this.#video.requestVideoFrameCallback((_now, metadata) =>
        run(metadata.width, metadata.height),
      );
      this.#cancelFrame = () => this.#video.cancelVideoFrameCallback(handle);
      return;
    }
    const handle = requestAnimationFrame(() =>
      run(this.#video.videoWidth, this.#video.videoHeight),
    );
    this.#cancelFrame = () => cancelAnimationFrame(handle);
  }

  #release(): void {
    this.#generation += 1;
    this.#starting = false;
    this.#cancelFrame?.();
    this.#cancelFrame = undefined;
    const stream = this.#stream;
    this.#stream = undefined;
    this.#video.pause();
    this.#video.srcObject = null;
    this.#video.hidden = true;
    this.#stopTracks(stream);
  }

  #stopTracks(stream: MediaStream | undefined): void {
    for (const track of stream?.getTracks() ?? []) {
      track.stop();
    }
  }

  #isCurrent(generation: number): boolean {
    return !this.#disposed && generation === this.#generation;
  }

  #isRunning(generation: number): boolean {
    return this.#isCurrent(generation) && this.#stream !== undefined;
  }
}
