/**
 * Port of the original's `dither`, `depthRamp`, `compress`, and `tonemap`.
 *
 * `depthRamp`'s three-way branch uses nested `select()` rather than `If/Else`: it is a
 * side-effect-free choice between three vec3 expressions.
 */
import {
  dot,
  float,
  fract,
  max,
  mix,
  pow,
  saturate,
  select,
  vec3,
} from 'three/tsl';
import type { Node } from 'three/webgpu';
import {
  HIGHLIGHT_BLEACH,
  LUMINANCE_WEIGHTS,
  WHITE_POINT,
} from './constants.ts';

type FloatNode = Node<'float'>;
type Vec2Node = Node<'vec2'>;
type Vec3Node = Node<'vec3'>;

/** Ordered-dither noise value in [0,1). */
export function dither(uv: Vec2Node): FloatNode {
  const point = uv.mul(1024);
  return fract(
    float(52.9829189).mul(
      fract(point.x.mul(0.06711056).add(point.y.mul(0.00583715))),
    ),
  ) as FloatNode;
}

const luminanceWeights = vec3(
  LUMINANCE_WEIGHTS[0],
  LUMINANCE_WEIGHTS[1],
  LUMINANCE_WEIGHTS[2],
);

const RAMP_COLD = vec3(0.03, 0.02, 0.12);
const RAMP_MIDDLE = vec3(0.11, 0.45, 0.94);
const RAMP_WARM = vec3(0.85, 0.36, 0.96);
const RAMP_HOT = vec3(0.97, 0.97, 0.87);

/** Cold-to-hot false-color ramp for the DEPTH view mode. */
export function depthRamp(value: FloatNode): Vec3Node {
  const low = mix(RAMP_COLD, RAMP_MIDDLE, value.div(0.4));
  const mid = mix(RAMP_MIDDLE, RAMP_WARM, value.sub(0.4).div(0.35));
  const high = mix(RAMP_WARM, RAMP_HOT, value.sub(0.75).div(0.25));
  return select(
    value.lessThan(0.4),
    low,
    select(value.lessThan(0.75), mid, high),
  ) as Vec3Node;
}

/** Reinhard-style luminance compression toward `WHITE_POINT`. */
export function compress(value: FloatNode): FloatNode {
  return value
    .mul(value.div(WHITE_POINT * WHITE_POINT).add(1))
    .div(value.add(1)) as FloatNode;
}

/** Full tonemap operator. Output is still linear-light. */
export function tonemap(color: Vec3Node): Vec3Node {
  const luminance = max(dot(color, luminanceWeights), 0.0001) as FloatNode;
  const mapped = compress(luminance);
  const shoulder = color.div(WHITE_POINT * WHITE_POINT).add(1);
  const perChannel = color.mul(shoulder).div(color.add(1));
  const bleach = pow(saturate(mapped), HIGHLIGHT_BLEACH);
  return saturate(
    mix(color.mul(mapped.div(luminance)), perChannel, bleach),
  ) as Vec3Node;
}
