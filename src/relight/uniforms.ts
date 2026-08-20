/**
 * Shared uniform nodes for the relighting pipeline, mirroring the original's
 * `RelightParams` and `DepthParams` structs. src/relight/ and src/renderer.ts import
 * these singletons instead of threading values through Fn arguments, matching the
 * original's single global bind-group layout.
 *
 * renderer.ts writes fresh values into `.value` each frame and each compute dispatch;
 * nothing here updates itself.
 */
import { Vector2, Vector4 } from 'three/webgpu';
import { uniform } from 'three/tsl';
import {
  DEFAULT_BULB_SIZE,
  DEFAULT_LIGHT_POSITION,
  DEFAULT_LIGHT_Z,
  RelightMode,
} from './constants.ts';

// --- Light ---
export const lightPositionUniform = uniform(
  new Vector2(DEFAULT_LIGHT_POSITION[0], DEFAULT_LIGHT_POSITION[1]),
);
export const lightZUniform = uniform(DEFAULT_LIGHT_Z);
export const lightColorUniform = uniform(new Vector4(1, 0.72, 0.46, 1));

// --- Look sliders (RelightSettings) ---
export const exposureUniform = uniform(0.5);
export const intensityUniform = uniform(3);
export const reliefUniform = uniform(0.85);
export const specularUniform = uniform(0.22);
export const shadowUniform = uniform(0.7);
export const occlusionUniform = uniform(0.55);
/** World-space radius of the emissive bulb disc (the original's fixed BULB_WORLD_RADIUS). */
export const bulbSizeUniform = uniform(DEFAULT_BULB_SIZE);

// --- Frame framing (per RenderFrame) ---
/** Row-major mat2 packed as [m00, m01, m10, m11], matching src/mat2.ts's `Mat2`. */
export const uvTransformUniform = uniform(new Vector4(1, 0, 0, 1));
export const sourceSizeUniform = uniform(new Vector2(1, 1));
export const mirrorUniform = uniform(true, 'bool');
export const swapAxesUniform = uniform(false, 'bool');

// --- View mode ---
export const modeUniform = uniform(RelightMode.RELIT, 'uint');

// --- Canvas framing ---
/**
 * The render target's width/height ratio. The lighting math runs in scene space,
 * `(uv - 0.5) * vec2(aspect, 1)`, so distances stay isotropic on non-square canvases:
 * circular light pool, undistorted bulb and shadows. The original always rendered a
 * square, so its uv space was already isotropic.
 */
export const canvasAspectUniform = uniform(1);

// --- Depth field (DepthParams-equivalent, consumed only by the compute passes) ---
/** (min, max) of the depth field's stabilized disparity range for this frame. */
export const depthRangeUniform = uniform(new Vector2(0, 1));
/** When true, depth-prepare writes the normalized value directly instead of blending history. */
export const resetUniform = uniform(true, 'bool');
/**
 * Temporal blend factors (constants.ts TEMPORAL_ALPHA / MOTION_ALPHA, written by
 * renderer.ts #writeBlendAlphas). The motion branch is rate-corrected for the actual
 * depth-update interval so moving subjects don't ghost; the low-motion branch keeps the
 * original's per-update constant, so per-inference noise is still smoothed away.
 */
export const temporalAlphaUniform = uniform(1);
export const motionAlphaUniform = uniform(1);
