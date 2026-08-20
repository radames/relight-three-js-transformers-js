/**
 * Port of the original's `cameraUvAt`.
 *
 * Maps a relit-scene UV to the camera source texture's UV, applying in order: mirroring,
 * a center "cover" crop to the canvas aspect (the original cropped to a square, its
 * render target always being one), and the caller-supplied `uvTransform` for
 * device-orientation correction, in src/mat2.ts's row-major convention
 * (`x' = m00*x + m01*y`, `y' = m10*x + m11*y`).
 *
 * The original derives `sourceSize` on-GPU with `textureDimensions()` on an external
 * texture. TSL has no equivalent that works for both `VideoTexture` and an
 * `ImageBitmap`-backed `Texture`, so renderer.ts writes the frame's `sourceSize` in
 * natural pixels into `sourceSizeUniform` each frame instead.
 */
import { select, vec2 } from 'three/tsl';
import type { Node } from 'three/webgpu';
import {
  canvasAspectUniform,
  mirrorUniform,
  sourceSizeUniform,
  swapAxesUniform,
  uvTransformUniform,
} from './uniforms.ts';

type Vec2Node = Node<'vec2'>;

export function cameraUvAt(uv: Vec2Node): Vec2Node {
  const sourceSize = select(
    swapAxesUniform,
    vec2(sourceSizeUniform.y, sourceSizeUniform.x),
    sourceSizeUniform,
  ) as Vec2Node;

  const framed = select(
    mirrorUniform,
    vec2(uv.x.oneMinus(), uv.y),
    uv,
  ) as Vec2Node;

  // Cover-crop the source to the canvas aspect (reduces to the original's
  // crop-to-square when aspect == 1).
  const cropHeight = sourceSize.y.min(sourceSize.x.div(canvasAspectUniform));
  const crop = vec2(cropHeight.mul(canvasAspectUniform), cropHeight);
  const sourcePixel = sourceSize
    .sub(crop)
    .mul(0.5)
    .add(framed.mul(crop))
    .sub(0.5);
  const clamped = sourcePixel.clamp(vec2(0), sourceSize.sub(1));
  const sourceUv = clamped.add(0.5).div(sourceSize);

  const centered = sourceUv.sub(0.5);
  const t = uvTransformUniform;
  const transformedX = t.x.mul(centered.x).add(t.y.mul(centered.y));
  const transformedY = t.z.mul(centered.x).add(t.w.mul(centered.y));
  return vec2(transformedX, transformedY).add(0.5);
}
