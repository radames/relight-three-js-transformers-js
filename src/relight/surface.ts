/**
 * Port of the original's `surfaceKernel`.
 *
 * Derives a per-pixel surface slope — a 7px-radius central-difference gradient softened
 * by `surfaceSlope` — and a two-radius, 8-neighbor height-field ambient occlusion term
 * from the normalized depth history buffer, then writes both plus the raw depth into an
 * RGBA storage texture the fragment pass samples.
 *
 * The original dispatches an 8x8 2D compute grid over `[width, height]`; this dispatches
 * a 1D grid over `width * height`, matching depth-prepare.ts, and derives `x`/`y` from
 * `instanceIndex`.
 */
import {
  Fn,
  float,
  instanceIndex,
  int,
  ivec2,
  max,
  saturate,
  textureStore,
  uint,
  vec2,
  vec4,
} from 'three/tsl';
import {
  HalfFloatType,
  LinearFilter,
  RGBAFormat,
  StorageTexture,
  type ComputeNode,
} from 'three/webgpu';
import {
  COMPUTE_WORKGROUP_SIZE,
  GRADIENT_BACK,
  GRADIENT_RADIUS,
  OCCLUSION_FLOOR,
  OCCLUSION_RADII,
  OCCLUSION_RANGE,
  OCCLUSION_SCALE,
  OCCLUSION_TAPS,
  RING_OFFSETS,
} from './constants.ts';
import {
  depthTexelAt,
  gentlerDelta,
  surfaceSlope,
  type FloatStorage,
} from './texel.ts';

export interface SurfaceResources {
  /** rgba16float storage texture: xy = surface slope, z = occlusion (1 = unoccluded), w = depth. */
  surfaceTexture: StorageTexture;
  /** The compute node to dispatch via `renderer.computeAsync(...)`. */
  computeNode: ComputeNode;
}

export function createSurface(
  history: FloatStorage,
  width: number,
  height: number,
): SurfaceResources {
  const pixelCount = width * height;

  const surfaceTexture = new StorageTexture(width, height);
  surfaceTexture.format = RGBAFormat;
  surfaceTexture.type = HalfFloatType;
  surfaceTexture.magFilter = LinearFilter;
  surfaceTexture.minFilter = LinearFilter;
  surfaceTexture.generateMipmaps = false;
  // `mipmapsAutoUpdate` exists at runtime but is missing from @types/three's
  // StorageTexture declaration, hence the cast.
  (
    surfaceTexture as unknown as { mipmapsAutoUpdate: boolean }
  ).mipmapsAutoUpdate = false;

  const kernel = Fn(() => {
    const index = instanceIndex;
    const x = int(index.mod(uint(width)));
    const y = int(index.div(uint(width)));

    const center = depthTexelAt(history, x, y, width, height);
    const left = depthTexelAt(history, x.add(GRADIENT_BACK), y, width, height);
    const right = depthTexelAt(
      history,
      x.add(GRADIENT_RADIUS),
      y,
      width,
      height,
    );
    const up = depthTexelAt(history, x, y.add(GRADIENT_BACK), width, height);
    const down = depthTexelAt(
      history,
      x,
      y.add(GRADIENT_RADIUS),
      width,
      height,
    );

    const rawGradient = vec2(
      gentlerDelta(center.sub(left), right.sub(center)),
      gentlerDelta(center.sub(up), down.sub(center)),
    ).div(GRADIENT_RADIUS);
    const gradient = surfaceSlope(rawGradient);

    const occlusion = float(0).toVar();
    // OCCLUSION_RADII and RING_OFFSETS are compile-time constants, so these nested JS
    // loops unroll into a fixed sequence of TSL expressions — the original's
    // `tgpu.unroll` — rather than emitting a runtime loop.
    for (const radius of OCCLUSION_RADII) {
      for (const stepY of RING_OFFSETS) {
        for (const stepX of RING_OFFSETS) {
          if (stepX !== 0 || stepY !== 0) {
            const neighbor = depthTexelAt(
              history,
              x.add(stepX * radius),
              y.add(stepY * radius),
              width,
              height,
            );
            const difference = neighbor.sub(center);
            const contact = float(1).sub(
              saturate(difference.abs().div(OCCLUSION_RANGE)),
            );
            const cleared = max(difference.sub(OCCLUSION_FLOOR), 0);
            occlusion.addAssign(
              saturate(cleared.div(OCCLUSION_SCALE)).mul(contact),
            );
          }
        }
      }
    }

    const value = vec4(
      gradient,
      float(1).sub(saturate(occlusion.div(OCCLUSION_TAPS))),
      center,
    );
    textureStore(surfaceTexture, ivec2(x, y), value);
  });

  const computeNode = kernel().compute(pixelCount, [COMPUTE_WORKGROUP_SIZE]);
  return { surfaceTexture, computeNode };
}
