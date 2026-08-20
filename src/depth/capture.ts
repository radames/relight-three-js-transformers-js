import { type Mat2 } from '../mat2.ts';
import type { RenderFrame } from '../renderer.ts';

/**
 * Builds the depth estimator's input, aligned pixel-for-pixel with what the renderer
 * displays. The depth texture must line up 1:1 with the visible camera image in canvas
 * UV space or the relighting swims relative to the picture.
 *
 * This mirrors `cameraUvAt` (camera-uv.ts): given a display UV, that function derives
 * the UV to sample the native (unrotated, unmirrored) source at, by cover-cropping to
 * the canvas aspect and applying a mirror flip and a 2x2 orientation transform
 * (`uvTransform`, only ever a rotation or reflection with entries in {-1, 0, 1}).
 *
 * That chain is affine, so rather than resampling pixel by pixel it is inverted into a
 * single matrix handed to `setTransform` before `drawImage`, letting the canvas
 * resample.
 *
 * The capture canvas shares the render target's aspect, with long side `baseSize`
 * quantized to multiples of 14 — Depth Anything's patch size, which also stops a 1px
 * window resize from rebuilding the GPU depth field. transformers.js returns the depth
 * map at the input canvas's own resolution, so it maps to screen UV 1:1.
 *
 * `baseSize` is the actual network resolution: both backends disable the processor's own
 * resize, so whatever is captured here is what the ViT runs on. Cost grows
 * superlinearly with it (patch attention is quadratic in patch count), making it the
 * main speed/quality dial and worth exposing in the GUI.
 *
 * Orientation: UV/canvas row 0 is the top, matching `RenderFrame.uvTransform`.
 */

const SIZE_QUANTUM = 14;

/**
 * Long-side pixel sizes offered for depth inference (all multiples of 14, Depth
 * Anything's ViT patch size). 518 is the model's native training resolution; the
 * smaller steps trade detail for real-time speed.
 */
export const INFERENCE_RESOLUTIONS = [252, 392, 448, 518] as const;
export type InferenceResolution = (typeof INFERENCE_RESOLUTIONS)[number];
/**
 * Matches the original example's fixed 448px DepthART resolution, so surface-slope
 * detail is comparable out of the box. Drop to 392/252 for speed, 518 for detail.
 */
export const DEFAULT_INFERENCE_RESOLUTION: InferenceResolution = 448;

/**
 * Capture dimensions for a canvas aspect and long-side `baseSize`. Exported because the
 * GPU path never draws a canvas but still needs the tensor shape this would produce.
 */
export function captureSize(
  aspect: number,
  baseSize: number,
): { width: number; height: number } {
  const quantize = (value: number): number =>
    Math.max(SIZE_QUANTUM, Math.round(value / SIZE_QUANTUM) * SIZE_QUANTUM);
  if (aspect >= 1) {
    return { width: baseSize, height: quantize(baseSize / aspect) };
  }
  return { width: quantize(baseSize * aspect), height: baseSize };
}

interface Affine {
  readonly a: number;
  readonly b: number;
  readonly c: number;
  readonly d: number;
  readonly e: number;
  readonly f: number;
}

const AFFINE_IDENTITY: Affine = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };

function scale(sx: number, sy: number): Affine {
  return { a: sx, b: 0, c: 0, d: sy, e: 0, f: 0 };
}

function translate(tx: number, ty: number): Affine {
  return { a: 1, b: 0, c: 0, d: 1, e: tx, f: ty };
}

/** The 2x2 linear map `x' = m00*x + m01*y; y' = m10*x + m11*y` as an affine matrix. */
function linear(m: Mat2): Affine {
  return { a: m[0], b: m[2], c: m[1], d: m[3], e: 0, f: 0 };
}

function invertMat2(m: Mat2): Mat2 {
  const det = m[0] * m[3] - m[1] * m[2];
  if (det === 0) {
    return m;
  }
  const invDet = 1 / det;
  return [m[3] * invDet, -m[1] * invDet, -m[2] * invDet, m[0] * invDet];
}

/** Composes two affines so that `apply(compose(outer, inner), p) === apply(outer, apply(inner, p))`. */
function compose(outer: Affine, inner: Affine): Affine {
  return {
    a: outer.a * inner.a + outer.c * inner.b,
    b: outer.b * inner.a + outer.d * inner.b,
    c: outer.a * inner.c + outer.c * inner.d,
    d: outer.b * inner.c + outer.d * inner.d,
    e: outer.a * inner.e + outer.c * inner.f + outer.e,
    f: outer.b * inner.e + outer.d * inner.f + outer.f,
  };
}

/**
 * Maps native source pixel space (unrotated, unmirrored) to destination canvas pixel
 * space: the exact inverse of `cameraUvAt`, so drawing the source through it reproduces
 * the crop, mirror and orientation the renderer shows.
 */
function forwardTransform(
  frame: Pick<
    RenderFrame,
    'sourceSize' | 'mirror' | 'uvTransform' | 'swapAxes'
  >,
  aspect: number,
  destWidth: number,
  destHeight: number,
): Affine {
  const { width, height } = frame.sourceSize;
  const swapped: readonly [number, number] = frame.swapAxes
    ? [height, width]
    : [width, height];
  // Cover-crop to the canvas aspect, matching camera-uv.ts exactly.
  const cropHeight = Math.min(swapped[1], swapped[0] / aspect);
  const cropWidth = cropHeight * aspect;
  const offsetX = (swapped[0] - cropWidth) / 2;
  const offsetY = (swapped[1] - cropHeight) / 2;

  // Inverted from cameraUvAt, composed in forward (native -> canvas) order:
  const toNativeUv = scale(1 / width, 1 / height);
  const uvTransformInverse = linear(invertMat2(frame.uvTransform));
  const uncenter = compose(
    translate(0.5, 0.5),
    compose(uvTransformInverse, translate(-0.5, -0.5)),
  );
  const toSwappedPixels = scale(swapped[0], swapped[1]);
  const toFramed = compose(
    translate(-offsetX / cropWidth, -offsetY / cropHeight),
    scale(1 / cropWidth, 1 / cropHeight),
  );
  const unmirror = frame.mirror
    ? compose(translate(1, 0), scale(-1, 1))
    : AFFINE_IDENTITY;
  const toCanvasPixels = scale(destWidth, destHeight);

  return compose(
    toCanvasPixels,
    compose(
      unmirror,
      compose(
        toFramed,
        compose(toSwappedPixels, compose(uncenter, toNativeUv)),
      ),
    ),
  );
}

let cachedCanvas: OffscreenCanvas | HTMLCanvasElement | undefined;
let cachedWidth = 0;
let cachedHeight = 0;

function getCanvas(
  width: number,
  height: number,
): OffscreenCanvas | HTMLCanvasElement {
  if (cachedCanvas && cachedWidth === width && cachedHeight === height) {
    return cachedCanvas;
  }
  cachedCanvas ??=
    typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(width, height)
      : document.createElement('canvas');
  cachedCanvas.width = width;
  cachedCanvas.height = height;
  cachedWidth = width;
  cachedHeight = height;
  return cachedCanvas;
}

function getContext2D(
  canvas: OffscreenCanvas | HTMLCanvasElement,
): OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D {
  const context = canvas.getContext('2d') as
    OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null;
  if (!context) {
    throw new Error(
      'Could not acquire a 2D canvas context for depth-inference capture.',
    );
  }
  return context;
}

/**
 * Draws `frame.source` into a reusable canvas, cover-cropped to `aspect` and oriented
 * like the on-screen render, ready as depth-estimator input.
 */
function captureForInference(
  frame: RenderFrame,
  aspect: number,
  baseSize: number = DEFAULT_INFERENCE_RESOLUTION,
): OffscreenCanvas | HTMLCanvasElement {
  const dest = captureSize(aspect, baseSize);
  const canvas = getCanvas(dest.width, dest.height);
  const ctx = getContext2D(canvas);

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, dest.width, dest.height);

  const { width, height } = frame.sourceSize;
  if (width <= 0 || height <= 0) {
    return canvas;
  }

  const t = forwardTransform(frame, aspect, dest.width, dest.height);
  ctx.setTransform(t.a, t.b, t.c, t.d, t.e, t.f);
  ctx.drawImage(frame.source, 0, 0, width, height);
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  return canvas;
}

/**
 * Like `captureForInference`, but returns an `ImageBitmap` transferable to the worker.
 * `transferToImageBitmap` leaves the canvas blank, which is fine: every capture starts
 * with a clear and redraw.
 */
export async function captureBitmapForInference(
  frame: RenderFrame,
  aspect: number,
  baseSize: number = DEFAULT_INFERENCE_RESOLUTION,
): Promise<ImageBitmap> {
  const canvas = captureForInference(frame, aspect, baseSize);
  if (canvas instanceof OffscreenCanvas) {
    return canvas.transferToImageBitmap();
  }
  return createImageBitmap(canvas);
}
