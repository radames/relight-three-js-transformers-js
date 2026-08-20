/**
 * Numeric constants ported verbatim from the original TypeGPU example's shaders.ts and
 * renderer.ts. Keep them in lock-step with the source; do not tidy the values.
 */

/** Ring of 8 neighbor offsets (3x3 minus center) used for occlusion + bulb sampling. */
export const RING_OFFSETS = [-1, 0, 1] as const;

/** Mirrors the original's `RelightMode`. The integer values are uploaded as a uniform. */
export const RelightMode = {
  RELIT: 0,
  CAMERA: 1,
  DEPTH: 2,
  NORMALS: 3,
} as const;

// RANGE_BLEND is omitted: percentile clipping and temporal range stabilization happen on
// the CPU before setDepthData(), replacing the original's stabilizeRangeKernel.

export const TEMPORAL_ALPHA = 0.32;
export const MOTION_ALPHA = 0.8;
export const MOTION_LOW = 0.02;
export const MOTION_HIGH = 0.09;

export const GRADIENT_RADIUS = 7;
export const GRADIENT_BACK = -GRADIENT_RADIUS;
export const GRADIENT_LIMIT = 0.009;
export const GRADIENT_NOISE = 0.0003;
export const GRADIENT_NOISE_ENERGY = GRADIENT_NOISE ** 2;
export const OCCLUSION_RADII = [3, 9] as const;
export const OCCLUSION_TAPS =
  OCCLUSION_RADII.length * (RING_OFFSETS.length ** 2 - 1);
export const OCCLUSION_SCALE = 0.07;
export const OCCLUSION_RANGE = 0.25;
export const OCCLUSION_FLOOR = 0.012;

export const NEAR_Z = 0;
/** Depth of the furthest surface the relit scene can hold. */
export const SURFACE_FAR_Z = -0.7;
export const LIGHT_RADIUS = 0.85;
export const LIGHT_WRAP = 0.25;
export const RELIEF_SCALE = 200;
export const SLOPE_COMPRESSION = 0.55;
export const SPECULAR_POWER = 36;
export const SPECULAR_F0 = 0.06;
export const GAMMA = 2.2;
export const WHITE_POINT = 2.6;
export const LUMINANCE_WEIGHTS: readonly [number, number, number] = [
  0.2126, 0.7152, 0.0722,
];
export const HIGHLIGHT_BLEACH = 2;
export const AMBIENT_FILL: readonly [number, number, number] = [0.78, 0.86, 1];
export const DITHER_STEP = 1 / 255;

/** Default for the "bulb size" setting, in place of the original's fixed 0.05 radius. */
export const DEFAULT_BULB_SIZE = 0.035;
export const BULB_CAMERA_Z = 2;
export const BULB_REFERENCE_Z = 0.42;
export const BULB_CORE = 8;
export const BULB_LIMB = 0.28;
export const BULB_EDGE = 0.75;
export const BULB_EDGE_FLOOR = 0.004;
export const BULB_EDGE_LIMIT = 0.3;
export const BULB_HALO = 1.6;
export const BULB_HALO_SPAN = 1.2;
export const BULB_VEIL = 0.12;
export const BULB_VEIL_SPAN = 4;
export const BULB_ONSET = 0.6;
export const BULB_OCCLUSION_SOFTNESS = 0.02;
export const BULB_SOURCE_SOFTNESS = 0.08;
export const BULB_SAMPLE_SPREAD = 0.6;
export const BULB_SAMPLES = RING_OFFSETS.length ** 2;
/**
 * The bulb's occlusion test reads a dedicated slow depth channel with a deadband
 * (depth-prepare.ts) rather than the fast field, so residual estimator noise cannot
 * flicker the bulb across its thin occlusion window. Deltas in normalized depth units
 * below LOW are treated as noise and creep in at BULB_DEPTH_CREEP per update; deltas
 * above HIGH — a real occluder — use the fast motion alpha.
 */
export const BULB_DEPTH_DEADBAND_LOW = 0.02;
export const BULB_DEPTH_DEADBAND_HIGH = 0.06;
export const BULB_DEPTH_CREEP = 0.04;

export const SHADOW_FAR_Z = -1.25;
export const SHADOW_STEPS = 32;
export const SHADOW_SPAN = 0.3;
export const SHADOW_BASELINE = 0.005;
export const SHADOW_BIAS = 0.014;
export const SHADOW_SLOPE_BIAS = 0.02;
export const SHADOW_THICKNESS = 0.7;
export const SHADOW_THICKNESS_GROWTH = 2.6;
export const SHADOW_SOFTNESS = 0.089;
export const SHADOW_GAIN = 2.5;
/** How far above the light plane an occluder may rise before it stops casting. */
export const SHADOW_FRONT_FADE = 0.2;

/**
 * The original's 1024 cap suited the docs site's small square viewer. This canvas fills
 * the viewport, where 1024 means rendering at ~1/3 of display resolution and letting the
 * browser stretch it. 2048 keeps fullscreen HiDPI output near-native while still bounding
 * the per-pixel ray-march cost.
 */
export const MAX_CANVAS_SIDE = 2048;
export const MAX_PIXEL_RATIO = 2;

const LIGHT_Z_CLEARANCE = 0.04;
export const LIGHT_Z_MIN = SURFACE_FAR_Z + LIGHT_Z_CLEARANCE;
export const LIGHT_Z_MAX = 1.65;

export const DEFAULT_LIGHT_POSITION: readonly [number, number] = [0.34, 0.34];
export const DEFAULT_LIGHT_Z = 0.42;

/** Workgroup size used for both 1D compute dispatches (depth-prepare and surface). */
export const COMPUTE_WORKGROUP_SIZE = 64;
