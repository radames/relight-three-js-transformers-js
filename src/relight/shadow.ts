/**
 * Port of the original's `depthAt` and `shadowFactor`.
 *
 * `shadowFactor` ray-marches `SHADOW_STEPS` samples along the light direction through the
 * deep, `SHADOW_FAR_Z`-based shadow-space height field, accumulating soft occlusion. It
 * uses a real TSL `Loop`/`If` rather than a JS unroll: 32 dependent texture samples is a
 * runtime loop, not a small compile-time unroll like the occlusion and bulb taps.
 */
import { If, Loop, float, texture, vec2 } from 'three/tsl';
import type { Node, Texture } from 'three/webgpu';
import {
  SHADOW_BASELINE,
  SHADOW_BIAS,
  SHADOW_FRONT_FADE,
  SHADOW_GAIN,
  SHADOW_SLOPE_BIAS,
  SHADOW_SOFTNESS,
  SHADOW_SPAN,
  SHADOW_STEPS,
  SHADOW_THICKNESS,
  SHADOW_THICKNESS_GROWTH,
} from './constants.ts';
import { canvasAspectUniform, lightZUniform } from './uniforms.ts';
import { shadowZ } from './texel.ts';

type FloatNode = Node<'float'>;
type Vec2Node = Node<'vec2'>;
type Vec3Node = Node<'vec3'>;

export interface ShadowModule {
  depthAt(uv: Vec2Node): FloatNode;
  shadowFactor(
    origin: Vec3Node,
    lightDirection: Vec3Node,
    reach: FloatNode,
    jitter: FloatNode,
  ): FloatNode;
}

export function createShadowModule(surfaceTexture: Texture): ShadowModule {
  function depthAt(uv: Vec2Node): FloatNode {
    return texture(surfaceTexture, uv).w as FloatNode;
  }

  function shadowFactor(
    origin: Vec3Node,
    lightDirection: Vec3Node,
    reach: FloatNode,
    jitter: FloatNode,
  ): FloatNode {
    // The march runs in aspect-scaled scene space (see relight-fragment.ts); sampling
    // the surface texture converts back to uv by undoing the x scale.
    const aspectVec = vec2(canvasAspectUniform, 1);
    const stride = reach.div(SHADOW_STEPS);
    const baselineTravel = reach.mul(SHADOW_BASELINE / SHADOW_SPAN);
    const trailProbe = origin.sub(lightDirection.mul(baselineTravel));
    const receiverRise = origin.z
      .sub(shadowZ(depthAt(trailProbe.xy.div(aspectVec).add(vec2(0.5)))))
      .sub(baselineTravel.mul(lightDirection.z))
      .max(0) as FloatNode;
    const risePerTravel = receiverRise.div(baselineTravel);

    const occlusion = float(0).toVar();
    Loop(SHADOW_STEPS, ({ i }) => {
      const travel = i.toFloat().add(jitter).mul(stride);
      const probe = origin.add(lightDirection.mul(travel));
      const sampleZ = shadowZ(depthAt(probe.xy.div(aspectVec).add(vec2(0.5))));
      const difference = sampleZ.sub(probe.z);
      const bias = float(SHADOW_BIAS).add(
        travel.mul(risePerTravel.add(SHADOW_SLOPE_BIAS)),
      );
      const thickness = float(SHADOW_THICKNESS).mul(
        float(1).add(travel.div(SHADOW_SPAN).mul(SHADOW_THICKNESS_GROWTH)),
      );

      If(
        difference.greaterThan(bias).and(difference.lessThan(thickness)),
        () => {
          const behindLight = float(1).sub(
            sampleZ.sub(lightZUniform).div(SHADOW_FRONT_FADE).saturate(),
          );
          occlusion.addAssign(
            difference
              .sub(bias)
              .div(SHADOW_SOFTNESS)
              .saturate()
              .mul(behindLight),
          );
        },
      );
    });

    return float(1).sub(
      occlusion.div(SHADOW_STEPS).mul(SHADOW_GAIN).saturate(),
    ) as FloatNode;
  }

  return { depthAt, shadowFactor };
}
