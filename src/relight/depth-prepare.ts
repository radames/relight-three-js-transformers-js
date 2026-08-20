/**
 * Port of the original's `depthPrepareKernel`.
 *
 * Reads raw relative disparity from a `DataTexture` (RedFormat/FloatType, one float per
 * pixel, uploaded by renderer.ts's `setDepthData`), normalizes it into [0,1] against the
 * already CPU-stabilized depth range uniform, and temporally smooths it against the
 * previous frame with a motion-adaptive blend factor.
 *
 * The original read a `vec4f` disparity storage buffer written by an on-GPU inference
 * pipeline; here the shell hands over plain disparity floats, so the source is a
 * texel-fetched DataTexture. `history` is still an `instancedArray` storage buffer.
 */
import {
  Fn,
  If,
  abs,
  float,
  instanceIndex,
  instancedArray,
  int,
  ivec2,
  max,
  mix,
  saturate,
  smoothstep,
  textureLoad,
  uint,
} from 'three/tsl';
import type { ComputeNode, DataTexture } from 'three/webgpu';
import {
  BULB_DEPTH_CREEP,
  BULB_DEPTH_DEADBAND_HIGH,
  BULB_DEPTH_DEADBAND_LOW,
  COMPUTE_WORKGROUP_SIZE,
  MOTION_HIGH,
  MOTION_LOW,
} from './constants.ts';
import {
  depthRangeUniform,
  motionAlphaUniform,
  resetUniform,
  temporalAlphaUniform,
} from './uniforms.ts';
import type { FloatStorage } from './texel.ts';

export interface DepthPrepareResources {
  /** Normalized, temporally-filtered depth field, one f32 per pixel (row-major). */
  history: FloatStorage;
  /**
   * A second, deliberately sluggish depth field read only by the bulb's occlusion test
   * (bulb.ts). Small per-update deltas — residual estimator noise — sit inside a deadband
   * and barely move it, so the bulb cannot flicker across its thin occlusion window;
   * large deltas, a real occluder arriving, still snap in at the fast motion alpha.
   */
  bulbHistory: FloatStorage;
  /** The compute node to dispatch via `renderer.computeAsync(...)`. */
  computeNode: ComputeNode;
}

export function createDepthPrepare(
  depthTexture: DataTexture,
  width: number,
  height: number,
): DepthPrepareResources {
  const pixelCount = width * height;
  const history = instancedArray(
    pixelCount,
    'float',
  ) as unknown as FloatStorage;
  const bulbHistory = instancedArray(
    pixelCount,
    'float',
  ) as unknown as FloatStorage;

  const kernel = Fn(() => {
    const index = instanceIndex;
    const x = int(index.mod(uint(width)));
    const y = int(index.div(uint(width)));
    const raw = textureLoad(depthTexture, ivec2(x, y)).r;

    const low = depthRangeUniform.x;
    const span = max(depthRangeUniform.y.sub(low), 0.001);

    const normalized = float(0).toVar();
    // NaN guard, mirroring the original's `disparity === disparity` check.
    If(raw.equal(raw), () => {
      normalized.assign(saturate(raw.sub(low).div(span)));
    });

    const filtered = float(normalized).toVar();
    const bulbFiltered = float(normalized).toVar();
    If(resetUniform, () => {
      filtered.assign(normalized);
      bulbFiltered.assign(normalized);
    }).Else(() => {
      const previous = history.element(index);
      const motion = smoothstep(
        MOTION_LOW,
        MOTION_HIGH,
        abs(normalized.sub(previous)),
      );
      // TEMPORAL_ALPHA/MOTION_ALPHA, rescaled by renderer.ts for the actual depth-update
      // rate — the original updated depth every frame, this does not.
      const alpha = mix(temporalAlphaUniform, motionAlphaUniform, motion);
      filtered.assign(mix(previous, normalized, alpha));

      // Slow bulb channel: deadband + hysteresis (see DepthPrepareResources doc).
      const bulbPrevious = bulbHistory.element(index);
      const gate = smoothstep(
        BULB_DEPTH_DEADBAND_LOW,
        BULB_DEPTH_DEADBAND_HIGH,
        abs(normalized.sub(bulbPrevious)),
      );
      const bulbAlpha = mix(float(BULB_DEPTH_CREEP), motionAlphaUniform, gate);
      bulbFiltered.assign(mix(bulbPrevious, normalized, bulbAlpha));
    });

    history.element(index).assign(filtered);
    bulbHistory.element(index).assign(bulbFiltered);
  });

  const computeNode = kernel().compute(pixelCount, [COMPUTE_WORKGROUP_SIZE]);
  return { history, bulbHistory, computeNode };
}
