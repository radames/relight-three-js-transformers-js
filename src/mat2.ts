/**
 * Minimal 2x2 matrix as a flat tuple, replacing TypeGPU's `d.mat2x2f`.
 * Layout is row-major: [m00, m01, m10, m11], applied as
 *   x' = m00 * x + m01 * y
 *   y' = m10 * x + m11 * y
 */
export type Mat2 = readonly [number, number, number, number];

export const IDENTITY_MAT2: Mat2 = [1, 0, 0, 1];

export function mat2Apply(m: Mat2, x: number, y: number): [number, number] {
  return [m[0] * x + m[1] * y, m[2] * x + m[3] * y];
}
