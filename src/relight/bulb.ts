/**
 * Port of the original's `bulbRadius`, `bulbExposure`, `bulbSurface`, `bulbGlow`, and
 * `bulbPresence`: the emissive disc drawn at the light's own position and depth, plus its
 * halo and veil falloff.
 *
 * `bulbExposure`'s 3x3 tap loop (`BULB_SAMPLES` = 9) is JS-unrolled at Fn-build time, like
 * the occlusion taps in surface.ts.
 *
 * Two differences from the original: the bulb's world radius is the adjustable
 * `bulbSizeUniform` rather than a constant, and every depth read here — the disc's
 * occlusion cut and the glow's 9 exposure probes — samples the slow, deadband-filtered
 * `bulbHistory` channel rather than the fast field, so estimator noise cannot flicker the
 * bulb across its thin occlusion window.
 */
import {
  exp,
  float,
  fwidth,
  length,
  max,
  mix,
  saturate,
  smoothstep,
  sqrt,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { Node } from 'three/webgpu';
import {
  BULB_CAMERA_Z,
  BULB_CORE,
  BULB_EDGE,
  BULB_EDGE_FLOOR,
  BULB_EDGE_LIMIT,
  BULB_HALO,
  BULB_HALO_SPAN,
  BULB_LIMB,
  BULB_ONSET,
  BULB_OCCLUSION_SOFTNESS,
  BULB_REFERENCE_Z,
  BULB_SAMPLES,
  BULB_SAMPLE_SPREAD,
  BULB_SOURCE_SOFTNESS,
  BULB_VEIL,
  BULB_VEIL_SPAN,
  RING_OFFSETS,
} from './constants.ts';
import {
  bulbSizeUniform,
  canvasAspectUniform,
  intensityUniform,
  lightPositionUniform,
  lightZUniform,
} from './uniforms.ts';
import { bilinearDepthAt, surfaceZ, type FloatStorage } from './texel.ts';

type FloatNode = Node<'float'>;
type Vec2Node = Node<'vec2'>;
type Vec3Node = Node<'vec3'>;
type Vec4Node = Node<'vec4'>;

export interface BulbModule {
  bulbSurface(uv: Vec2Node, tint: Vec3Node): Vec4Node;
  bulbGlow(uv: Vec2Node, tint: Vec3Node): Vec3Node;
  bulbPresence(): FloatNode;
}

export function createBulbModule(
  bulbDepth: FloatStorage,
  width: number,
  height: number,
): BulbModule {
  function depthAt(uv: Vec2Node): FloatNode {
    return bilinearDepthAt(bulbDepth, uv, width, height);
  }

  function bulbRadius(): FloatNode {
    return bulbSizeUniform
      .mul(BULB_CAMERA_Z - BULB_REFERENCE_Z)
      .div(float(BULB_CAMERA_Z).sub(lightZUniform)) as FloatNode;
  }

  function bulbExposure(radius: FloatNode): FloatNode {
    // Probe offsets are radii in scene space; dividing by (aspect, 1) makes them uv-space
    // offsets, keeping the sampled ring circular on screen.
    const aspectVec = vec2(canvasAspectUniform, 1);
    const open = float(0).toVar();
    for (const stepY of RING_OFFSETS) {
      for (const stepX of RING_OFFSETS) {
        const probe = lightPositionUniform.add(
          vec2(stepX, stepY).mul(radius.mul(BULB_SAMPLE_SPREAD)).div(aspectVec),
        );
        open.addAssign(
          smoothstep(
            0,
            BULB_SOURCE_SOFTNESS,
            lightZUniform.sub(surfaceZ(depthAt(probe))),
          ),
        );
      }
    }
    return open.div(BULB_SAMPLES) as FloatNode;
  }

  function bulbSurface(uv: Vec2Node, tint: Vec3Node): Vec4Node {
    const radius = bulbRadius();
    // Distance measured in aspect-scaled scene space so the disc stays circular.
    const spread = length(
      uv.sub(lightPositionUniform).mul(vec2(canvasAspectUniform, 1)),
    ).div(radius) as FloatNode;
    const limb = saturate(spread);
    const dome = sqrt(max(float(1).sub(limb.mul(limb)), 0)) as FloatNode;
    const facing = dome.mul(dome) as FloatNode;
    const front = lightZUniform.add(dome.mul(bulbSizeUniform));
    const solid = smoothstep(
      0,
      BULB_OCCLUSION_SOFTNESS,
      front.sub(surfaceZ(depthAt(uv))),
    );
    const edge = fwidth(spread)
      .mul(BULB_EDGE)
      .clamp(BULB_EDGE_FLOOR, BULB_EDGE_LIMIT);
    const coverage = float(1)
      .sub(smoothstep(float(1).sub(edge), float(1).add(edge), spread))
      .mul(solid);
    const hue = mix(tint, vec3(1), facing.mul(facing));
    return vec4(
      hue.mul(mix(float(BULB_LIMB), float(1), facing).mul(BULB_CORE)),
      coverage,
    ) as Vec4Node;
  }

  function bulbGlow(uv: Vec2Node, tint: Vec3Node): Vec3Node {
    const radius = bulbRadius();
    const radii = length(
      uv.sub(lightPositionUniform).mul(vec2(canvasAspectUniform, 1)),
    ).div(radius) as FloatNode;
    const halo = exp(radii.div(BULB_HALO_SPAN).negate());
    const veil = exp(radii.div(BULB_VEIL_SPAN).negate());
    return tint.mul(
      halo.mul(BULB_HALO).add(veil.mul(BULB_VEIL)).mul(bulbExposure(radius)),
    ) as Vec3Node;
  }

  function bulbPresence(): FloatNode {
    return saturate(intensityUniform.div(BULB_ONSET)) as FloatNode;
  }

  return { bulbSurface, bulbGlow, bulbPresence };
}
