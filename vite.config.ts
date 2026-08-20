import { defineConfig } from 'vite';

export default defineConfig({
  // Relative base so the built site works from any path: the repo root in dev,
  // /relight-three-js-transformers-js/ on GitHub Pages, or a Space subpath.
  base: './',
  build: {
    // The transformers.js/onnxruntime bundle is large by nature; don't warn about it.
    chunkSizeWarningLimit: 2000,
  },
});
