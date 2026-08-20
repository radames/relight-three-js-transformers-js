/**
 * CPU-side replacement for the original's `stabilizeRangeKernel` compute pass: finds a
 * robust [min, max] window of a raw depth frame by percentile clipping, then smooths that
 * window across frames so the relighting doesn't flicker as the estimate jitters.
 */

export interface Range {
  readonly min: number;
  readonly max: number;
}

function swap(arr: Float32Array, i: number, j: number): void {
  const tmp = arr[i] as number;
  arr[i] = arr[j] as number;
  arr[j] = tmp;
}

/**
 * In-place order-statistic selection (quickselect, Lomuto partition, random pivot), O(n)
 * average. `arr[k]` ends up holding the k-th smallest value in `arr[lo..hi]`, everything
 * left of it `<=` and everything right `>=`, so a later call with a bigger `k` can reuse
 * the same array and a narrowed `[lo, hi]`.
 */
function quickselect(
  arr: Float32Array,
  k: number,
  lo: number,
  hi: number,
): number {
  let low = lo;
  let high = hi;
  while (low < high) {
    const pivotIndex = low + Math.floor(Math.random() * (high - low + 1));
    swap(arr, pivotIndex, high);
    const pivot = arr[high] as number;
    let store = low;
    for (let i = low; i < high; i++) {
      if ((arr[i] as number) < pivot) {
        swap(arr, i, store);
        store++;
      }
    }
    swap(arr, store, high);
    if (store === k) {
      return arr[k] as number;
    }
    if (store < k) {
      low = store + 1;
    } else {
      high = store - 1;
    }
  }
  return arr[low] as number;
}

/**
 * Finds the [lo, hi] percentile window of `data`, ignoring non-finite entries (the network
 * can leave NaNs at invalid pixels). Quickselects over a private copy, never a full sort.
 */
export function percentileRange(
  data: Float32Array,
  lo = 0.02,
  hi = 0.98,
): Range {
  const n = data.length;
  if (n === 0) {
    return { min: 0, max: 1 };
  }

  const work = new Float32Array(n);
  let count = 0;
  for (let i = 0; i < n; i++) {
    const value = data[i] as number;
    if (Number.isFinite(value)) {
      work[count++] = value;
    }
  }
  if (count === 0) {
    return { min: 0, max: 1 };
  }

  const loIndex = Math.min(
    count - 1,
    Math.max(0, Math.floor(lo * (count - 1))),
  );
  const hiIndex = Math.min(
    count - 1,
    Math.max(loIndex, Math.ceil(hi * (count - 1))),
  );

  const min = quickselect(work, loIndex, 0, count - 1);
  const max =
    loIndex === hiIndex ? min : quickselect(work, hiIndex, loIndex, count - 1);

  return { min, max: Math.max(max, min + 0.001) };
}

/** How fast the aligner's reference frame follows the raw stream, per update. */
const ALIGN_REF_BLEND = 0.05;
/** Degenerate-frame guard: per-frame global rescale is clamped to this factor. */
const ALIGN_SCALE_LIMIT = 4;

/**
 * Removes the network's global scale/offset flicker. Relative-depth models wobble the
 * whole map by a few percent frame to frame even for a static scene, and that global term
 * dominates the temporal instability downstream (bulb occlusion, shadows, normals).
 *
 * Each frame is affinely remapped in place (`x * scale + offset`) so its median and
 * mean-absolute-deviation match a slowly-moving reference — the scale/shift alignment
 * trick from video-depth work. Relative structure passes through untouched. NaNs are
 * skipped when measuring and preserved by the remap.
 */
export class DepthAligner {
  #referenceMedian: number | undefined;
  #referenceSpread = 1;

  align(data: Float32Array): void {
    const n = data.length;
    const work = new Float32Array(n);
    let count = 0;
    for (let i = 0; i < n; i++) {
      const value = data[i] as number;
      if (Number.isFinite(value)) {
        work[count++] = value;
      }
    }
    if (count === 0) {
      return;
    }
    const frameMedian = quickselect(work, count >> 1, 0, count - 1);
    let deviation = 0;
    for (let i = 0; i < count; i++) {
      deviation += Math.abs((work[i] as number) - frameMedian);
    }
    const frameSpread = Math.max(deviation / count, 1e-6);

    const referenceMedian = this.#referenceMedian;
    if (referenceMedian === undefined) {
      this.#referenceMedian = frameMedian;
      this.#referenceSpread = frameSpread;
      return;
    }
    const scale = Math.min(
      Math.max(this.#referenceSpread / frameSpread, 1 / ALIGN_SCALE_LIMIT),
      ALIGN_SCALE_LIMIT,
    );
    const offset = referenceMedian - frameMedian * scale;
    for (let i = 0; i < n; i++) {
      data[i] = (data[i] as number) * scale + offset;
    }
    this.#referenceMedian = mix(referenceMedian, frameMedian, ALIGN_REF_BLEND);
    this.#referenceSpread = mix(
      this.#referenceSpread,
      frameSpread,
      ALIGN_REF_BLEND,
    );
  }

  reset(): void {
    this.#referenceMedian = undefined;
    this.#referenceSpread = 1;
  }
}

/** Catch-up rate per update once the median target leaves the deadband. */
const RANGE_BLEND = 0.2;
/** Recentering rate per update while the target sits inside the deadband. */
const RANGE_CREEP = 0.01;
/**
 * Deviation, as a fraction of the stable range's span, treated as estimator noise.
 * Deviations under this bound must not move the normalization window, or every stationary
 * object's normalized z drifts with the network's frame-to-frame wobble.
 */
const RANGE_DEADBAND = 0.06;
/** Per-frame ranges are median-filtered over this many updates before being chased. */
const MEDIAN_WINDOW = 9;

function mix(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2
    ? (sorted[mid] as number)
    : ((sorted[mid - 1] as number) + (sorted[mid] as number)) / 2;
}

/**
 * Temporally stabilizes a sequence of per-frame ranges. The original's
 * `stabilizeRangeKernel` eased toward every frame's range, which let the normalization
 * window drift continuously with estimator noise. This version holds it still:
 *
 * 1. Per-frame ranges are median-filtered over the last `MEDIAN_WINDOW` updates,
 *    discarding single-frame outliers.
 * 2. The stable range chases that median with hysteresis: inside `RANGE_DEADBAND` it only
 *    creeps at `RANGE_CREEP`, and only a sustained deviation beyond it — a real scene
 *    change — triggers the fast `RANGE_BLEND`.
 *
 * On the frame right after `reset()`, the stable range snaps to the frame's range.
 */
export class RangeStabilizer {
  #stable: Range | undefined;
  #minHistory: number[] = [];
  #maxHistory: number[] = [];

  next(frameRange: Range): Range {
    this.#minHistory.push(frameRange.min);
    this.#maxHistory.push(frameRange.max);
    if (this.#minHistory.length > MEDIAN_WINDOW) {
      this.#minHistory.shift();
      this.#maxHistory.shift();
    }
    const target: Range = {
      min: median(this.#minHistory),
      max: median(this.#maxHistory),
    };
    const stable = this.#stable;
    if (!stable) {
      this.#stable = { ...target };
      return this.#stable;
    }
    const span = Math.max(stable.max - stable.min, 1e-6);
    const rate = (current: number, next: number): number =>
      Math.abs(next - current) > span * RANGE_DEADBAND
        ? RANGE_BLEND
        : RANGE_CREEP;
    this.#stable = {
      min: mix(stable.min, target.min, rate(stable.min, target.min)),
      max: mix(stable.max, target.max, rate(stable.max, target.max)),
    };
    return this.#stable;
  }

  reset(): void {
    this.#stable = undefined;
    this.#minHistory = [];
    this.#maxHistory = [];
  }
}
