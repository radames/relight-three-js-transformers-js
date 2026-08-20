/**
 * Port of the original's `relightFragment`, the entry point tying every other
 * src/relight/ module together into the final display color.
 *
 * TSL has no value-carrying early return inside nested `If` blocks — `Return()` emits a
 * bare `return;`. The original's four-mode early-return chain is therefore a single
 * mutable `output` var assigned inside nested `If(...).Else(...)` blocks, the innermost
 * `Else` holding the full relit-lighting path. Mode semantics are unchanged; the branch
 * has to be TSL's `If`/`Else` rather than a JS `if` because the mode is a uniform.
 */
import {
  Fn,
  If,
  dot,
  float,
  length,
  max,
  mix,
  normalize,
  pow,
  saturate,
  texture,
  uv as screenUv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { Node, StorageTexture, Texture } from 'three/webgpu';
import {
  AMBIENT_FILL,
  DITHER_STEP,
  GAMMA,
  LIGHT_RADIUS,
  LIGHT_WRAP,
  RELIEF_SCALE,
  RelightMode,
  SHADOW_SPAN,
  SLOPE_COMPRESSION,
  SPECULAR_F0,
  SPECULAR_POWER,
} from './constants.ts';
import { shadowZ, surfaceZ, type FloatStorage } from './texel.ts';
import { cameraUvAt } from './camera-uv.ts';
import { dither, depthRamp, tonemap } from './tonemap.ts';
import { createShadowModule } from './shadow.ts';
import { createBulbModule } from './bulb.ts';
import {
  canvasAspectUniform,
  exposureUniform,
  intensityUniform,
  lightColorUniform,
  lightPositionUniform,
  lightZUniform,
  modeUniform,
  occlusionUniform,
  reliefUniform,
  shadowUniform,
  specularUniform,
} from './uniforms.ts';

type TextureNode = ReturnType<typeof texture>;

export interface RelightFragmentResources {
  /** The full-screen relit color, suitable as a `NodeMaterial.fragmentNode`. */
  fragmentNode: Node<'vec4'>;
  /**
   * The camera source's texture node. renderer.ts swaps `.value` between a `VideoTexture`
   * and an `ImageBitmap`-backed `Texture` without rebuilding the node graph.
   */
  cameraTextureNode: TextureNode;
}

const ambientFillTint = vec3(AMBIENT_FILL[0], AMBIENT_FILL[1], AMBIENT_FILL[2]);

export function createRelightFragment(
  surfaceTexture: StorageTexture,
  bulbDepth: FloatStorage,
  width: number,
  height: number,
  initialCameraTexture: Texture,
): RelightFragmentResources {
  const cameraTextureNode = texture(initialCameraTexture);
  const shadowModule = createShadowModule(surfaceTexture);
  const bulbModule = createBulbModule(bulbDepth, width, height);

  const fragmentNode = Fn(() => {
    const uv = screenUv();
    const cameraColor = saturate(cameraTextureNode.sample(cameraUvAt(uv)).rgb);

    const output = vec4(0, 0, 0, 1).toVar();

    If(modeUniform.equal(RelightMode.CAMERA), () => {
      output.assign(vec4(cameraColor, 1));
    }).Else(() => {
      const surface = texture(surfaceTexture, uv);

      If(modeUniform.equal(RelightMode.DEPTH), () => {
        output.assign(vec4(depthRamp(saturate(surface.w)), 1));
      }).Else(() => {
        const slope = surface.xy.mul(reliefUniform.mul(RELIEF_SCALE));
        const tilt = vec2(0, 0).sub(
          slope.div(float(1).add(length(slope).mul(SLOPE_COMPRESSION))),
        );
        const normal = normalize(vec3(tilt, 1));

        If(modeUniform.equal(RelightMode.NORMALS), () => {
          output.assign(vec4(normal.mul(0.5).add(0.5), 1));
        }).Else(() => {
          // --- Full relit lighting path (RelightMode.RELIT) ---
          // Lighting runs in scene space, (uv - 0.5) * (aspect, 1), so distances are
          // isotropic on non-square canvases (see canvasAspectUniform).
          const aspectVec = vec2(canvasAspectUniform, 1);
          const centered = uv.sub(0.5).mul(aspectVec);
          const noise = dither(uv);
          const position = vec3(centered, surfaceZ(surface.w));
          const lightPosition3 = vec3(
            lightPositionUniform.sub(0.5).mul(aspectVec),
            lightZUniform,
          );
          const toLight = lightPosition3.sub(position);
          const distance = max(length(toLight), 0.0001);
          const lightDirection = toLight.div(distance);
          const spread = distance.div(LIGHT_RADIUS);
          const falloff = float(1).div(float(1).add(spread.mul(spread)));
          const wrapped = saturate(
            dot(normal, lightDirection)
              .add(LIGHT_WRAP)
              .div(1 + LIGHT_WRAP),
          );
          const lambert = wrapped.mul(wrapped);

          const shadow = float(1).toVar();
          If(shadowUniform.greaterThan(0), () => {
            const shadowOrigin = vec3(centered, shadowZ(surface.w));
            const shadowToLight = lightPosition3.sub(shadowOrigin);
            const shadowDistance = max(length(shadowToLight), 0.0001);
            const reach = shadowDistance.mul(
              float(SHADOW_SPAN).div(
                max(length(shadowToLight.xy), SHADOW_SPAN),
              ),
            );
            const traced = shadowModule.shadowFactor(
              shadowOrigin,
              shadowToLight.div(shadowDistance),
              reach,
              noise,
            );
            shadow.assign(mix(float(1), traced, shadowUniform));
          });

          const occlusion = mix(float(1), surface.z, occlusionUniform);

          const albedo = pow(cameraColor, vec3(GAMMA));
          const tint = lightColorUniform.rgb;
          const halfDirection = normalize(lightDirection.add(vec3(0, 0, 1)));
          const lobe = pow(
            saturate(dot(normal, halfDirection)),
            SPECULAR_POWER,
          );
          const grazing = pow(float(1).sub(saturate(normal.z)), 5);
          const highlight = lobe.mul(
            float(SPECULAR_F0).add(float(1 - SPECULAR_F0).mul(grazing)),
          );

          const lit = albedo
            .mul(ambientFillTint)
            .mul(exposureUniform.mul(occlusion))
            .toVar();
          lit.addAssign(
            albedo
              .mul(tint)
              .mul(lambert.mul(falloff).mul(shadow).mul(intensityUniform)),
          );
          lit.addAssign(
            tint.mul(
              highlight
                .mul(falloff)
                .mul(shadow)
                .mul(occlusion)
                .mul(specularUniform)
                .mul(intensityUniform),
            ),
          );

          const presence = bulbModule.bulbPresence();
          const bulb = bulbModule.bulbSurface(uv, tint);
          lit.assign(mix(lit, bulb.xyz.mul(presence), bulb.w.mul(presence)));
          lit.addAssign(bulbModule.bulbGlow(uv, tint).mul(presence));

          const display = pow(tonemap(lit), vec3(1 / GAMMA));
          output.assign(vec4(display.add(noise.sub(0.5).mul(DITHER_STEP)), 1));
        });
      });
    });

    return output;
  })();

  return { fragmentNode, cameraTextureNode };
}
