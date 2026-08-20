/**
 * Depth-field texel helpers, ported from the original's `texelIndex` / `depthTexelAt` /
 * `gentlerDelta` / `surfaceSlope` / `surfaceZ` / `shadowZ`.
 *
 * `history` (an `instancedArray`-backed storage buffer, one f32 per pixel) stands in for
 * the original's `surfaceLayout.$.depth` storage array. Width/height are plain JS numbers
 * baked in at Fn-build time — the compute graphs are rebuilt whenever the depth field's
 * dimensions change — matching the original's fixed-size dispatch.
 *
 * Helpers here are typed with concrete node-type literals rather than `Node<any>`:
 * `Node<any>`'s conditional branch (`unknown extends TNodeType ? {} : ...`) collapses to
 * `{}` for `any`, silently stripping every chained method (`.mul()`, `.sub()`, ...) from
 * the type.
 */
import {
  abs,
  clamp,
  floor,
  int,
  length,
  max,
  mix,
  sqrt,
  tanh,
  uint,
  vec2,
} from 'three/tsl';
import type { Node } from 'three/webgpu';
import {
  GRADIENT_LIMIT,
  GRADIENT_NOISE_ENERGY,
  NEAR_Z,
  SURFACE_FAR_Z,
  SHADOW_FAR_Z,
} from './constants.ts';

type FloatNode = Node<'float'>;
type IntNode = Node<'int'>;
type UintNode = Node<'uint'>;
type Vec2Node = Node<'vec2'>;

/**
 * TSL's `clamp()` only declares float/vec2/vec3/vec4 overloads. Int clamping works at
 * runtime — WGSL has no restriction — so this alias re-types the same function.
 */
const clampInt = clamp as unknown as (
  value: IntNode,
  low: IntNode,
  high: IntNode,
) => IntNode;

/** A storage buffer node produced by `instancedArray(count, 'float')`. */
export interface FloatStorage {
  element(index: UintNode | IntNode): FloatNode;
}

/** Clamped row-major index into a `width x height` field, matching the original's `texelIndex`. */
export function texelIndex(
  x: IntNode,
  y: IntNode,
  width: number,
  height: number,
): UintNode {
  const clampedX = clampInt(x, int(0), int(width - 1));
  const clampedY = clampInt(y, int(0), int(height - 1));
  return uint(clampedY.mul(int(width)).add(clampedX)) as UintNode;
}

/** Clamped indexed read of the depth history field, matching the original's `depthTexelAt`. */
export function depthTexelAt(
  history: FloatStorage,
  x: IntNode,
  y: IntNode,
  width: number,
  height: number,
): FloatNode {
  return history.element(texelIndex(x, y, width, height));
}

/**
 * Bilinear sample of a depth history field at a [0,1]² uv, texel centers at
 * (i + 0.5) / size. Storage buffers have no sampler, so the filtering `texture(...)`
 * would give for free is done with four clamped texel reads.
 */
export function bilinearDepthAt(
  history: FloatStorage,
  uv: Vec2Node,
  width: number,
  height: number,
): FloatNode {
  const position = uv.mul(vec2(width, height)).sub(vec2(0.5, 0.5));
  const base = floor(position);
  const frac = position.sub(base);
  const x0 = int(base.x);
  const y0 = int(base.y);
  const x1 = x0.add(int(1));
  const y1 = y0.add(int(1));
  const top = mix(
    depthTexelAt(history, x0, y0, width, height),
    depthTexelAt(history, x1, y0, width, height),
    frac.x,
  );
  const bottom = mix(
    depthTexelAt(history, x0, y1, width, height),
    depthTexelAt(history, x1, y1, width, height),
    frac.x,
  );
  return mix(top, bottom, frac.y) as FloatNode;
}

/** Central-difference-ish derivative that favors the smaller-magnitude side, avoiding edge blowups. */
export function gentlerDelta(
  backward: FloatNode,
  forward: FloatNode,
): FloatNode {
  const back = abs(backward) as FloatNode;
  const front = abs(forward) as FloatNode;
  return backward
    .mul(front)
    .add(forward.mul(back))
    .div(max(back.add(front), 1e-9)) as FloatNode;
}

/** Soft-clamps a raw gradient to a tanh ceiling to tame noise-driven spikes. */
export function surfaceSlope(gradient: Vec2Node): Vec2Node {
  const steepness = max(length(gradient), 1e-9) as FloatNode;
  const shrunk = sqrt(
    max(steepness.mul(steepness).sub(GRADIENT_NOISE_ENERGY), 0),
  ) as FloatNode;
  const ceiling = tanh(shrunk.div(GRADIENT_LIMIT)).mul(
    GRADIENT_LIMIT,
  ) as FloatNode;
  return gradient.mul(ceiling.div(steepness)) as Vec2Node;
}

/** Maps normalized depth [0,1] to the relit scene's world-space Z range. */
export function surfaceZ(depth: FloatNode): FloatNode {
  return mix(SURFACE_FAR_Z, NEAR_Z, depth) as FloatNode;
}

/** Maps normalized depth [0,1] to the (deeper) world-space Z range used for shadow ray marching. */
export function shadowZ(depth: FloatNode): FloatNode {
  return mix(SHADOW_FAR_Z, NEAR_Z, depth) as FloatNode;
}
