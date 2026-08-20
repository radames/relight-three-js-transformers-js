https://github.com/user-attachments/assets/8df51ae6-e650-46c5-a809-907daa0cae40

# Monocular Light Injection — Three.js + Transformers.js

**Live demo:** <https://radames.github.io/relight-three-js-transformers-js/>
(needs a WebGPU-capable browser)

A standalone Three.js (WebGPU + TSL) port of the TypeGPU
[monocular-light-injection example](https://docs.swmansion.com/TypeGPU/examples/#example=image-processing--monocular-light-injection):
take a webcam feed or a photo, estimate per-pixel depth with a neural network in the
browser, and inject a movable virtual point light into the scene, with depth-aware
diffuse/specular shading, ray-marched soft shadows, and a glowing bulb sprite.

The original runs a custom DepthART network hand-written as TypeGPU compute shaders.
This port has **no TypeGPU dependency**:

- **Rendering** — Three.js `WebGPURenderer` with TSL node shaders: a temporal
  depth-stabilization compute pass, a surface (slope + ambient occlusion) compute pass
  writing to a storage texture, and a fullscreen relighting fragment (wrapped-Lambert
  diffuse, Blinn-Phong-style specular, 32-step ray-marched shadows, bulb sprite,
  Reinhard-extended tonemap + dither). See `src/relight/`. A single
  `requestAnimationFrame` loop renders at the display's refresh rate for both sources;
  camera frames only swap in the latest video frame, so a 30 fps webcam does not cap
  the frame rate (the relight pass itself costs ~1–3 ms of GPU time per frame).
- **Depth inference** — [Depth Anything V2](https://huggingface.co/onnx-community/depth-anything-v2-small)
  (small / base / large, selectable in the chooser) on the WebGPU backend of
  [transformers.js](https://github.com/huggingface/transformers.js), in `fp16` where the
  device supports `shader-f16`. The network runs at the selectable "depth res"
  (252 / 392 / 448 / 518 px long side); per-frame timing shows in the panel's read-only
  "depth" row. The ONNX Runtime execution provider is switchable at runtime from the
  panel's **engine** dropdown (`webgpu`, `wasm`, plus `webnn-gpu` / `webnn-npu` where the
  browser exposes WebNN), and the read-only **backend** row reports what actually loaded
  (e.g. `webgpu · fp16 · shared device`, `wasm · q8`). See `src/depth/`.

  The engine picks one of two backends; there is no separate knob.

  - **`webgpu` (default) — shared device.** ONNX Runtime and Three.js share one
    `GPUDevice`, so the frame never leaves the GPU on its way into the network: a compute
    pass samples the video texture with the same framing the renderer uses and writes the
    normalized NCHW tensor straight into a storage buffer, which the session reads in
    place. That removes the per-frame 2D-canvas draw, the `getImageData` readback, and the
    image processor's CPU rescale/normalize/pack — measured at ~39% off depth latency
    (245 ms → 150 ms for Depth Anything V2 small, fp16, 448 px, live camera, headless
    Chrome on Apple Silicon; a relative figure, not an absolute hardware one).

    A `GPUBuffer` cannot cross a thread boundary, so this backend runs on the main thread;
    what is left there is a dispatch and an async `session.run`. That is the trade: the
    depth field is fresher, but ONNX Runtime now dispatches its kernels on the main thread
    and the render loop gets less of it. ONNX Runtime accepts an adapter but never a
    device, so it always creates its own and the renderer borrows it — which is why the
    model loads before the renderer is built.

    Sharing is best-effort. The runtime releases its device along with its last WebGPU
    session, so reloading the model hands back a fresh one the already-built renderer
    cannot move to. The two devices then run side by side, the input-side win intact
    because the compute pass never touched the renderer's device, and the backend row
    reads `gpu input` instead of `shared device`.

  - **`wasm` / `webnn-gpu` / `webnn-npu` — worker.** No device to share, so inference goes
    back to the original path: a 2D-canvas capture is transferred to a Web Worker running
    the full transformers.js pipeline, which transfers a `Float32Array` back. Same model,
    resolution, and alignment maths. It is also the automatic fallback whenever the
    shared-device path cannot come up.

  The panel's "depth" row times the whole frame-to-prediction span on the main thread for
  both backends, capture and the percentile scan included, so it reads as the real
  depth-update period rather than just the model call.

  Borrowing a device also means owning its lifetime. ONNX Runtime destroys its `GPUDevice`
  as soon as its last WebGPU session is released, which would take the renderer's canvas
  down with it on an engine or model switch (`The Device was lost` on every frame after).
  `borrowDevice` in `src/renderer.ts` neutralizes `destroy` on a device three.js did not
  create; the browser still reclaims it on unload.

  One caveat on a borrowed device: three.js requests every adapter feature when it makes
  its own device, but can only use what the owner asked for. Nothing here needs an
  optional feature — the depth field is texel-fetched `r32float`, so `float32-filterable`
  never comes up — except GPU timestamps, so the panel's render "ms gpu" figure may read
  as unavailable on the `webgpu` engine when the runtime's device lacks `timestamp-query`.

- **UI** — [lil-gui](https://lil-gui.georgealways.com/) panel + a small custom
  chooser (live camera / demo photo / your photo, and depth model size).

## Run

Requires Node 20.19+ or 22.12+ (Vite 8); with nvm, `nvm use` picks it up from `.nvmrc`.

```sh
npm install --ignore-scripts   # --ignore-scripts skips sharp, a Node-only transformers.js dep unused in the browser
npm run dev
```

Open http://localhost:5173 in a WebGPU-capable browser (Chrome/Edge 121+). The
depth model downloads from the Hugging Face Hub on first start and is cached by the
browser afterwards (small ≈ 25 MB, base ≈ 100 MB, large ≈ 330 MB in fp16).

## Deploying

`.github/workflows/deploy.yml` builds the site with `npm run build` and publishes
`dist/` to GitHub Pages on every push to `main` (Pages source: GitHub Actions). The
Vite `base` is relative, so the build also works from any other subpath.

## Controls

- **auto orbit** (panel, off by default) — check it and the light circles on its own,
  ignoring pointer input. Left off, you steer the light:
  - **Hover** — light follows the cursor; **click/tap** — pin/unpin the light.
  - **Wheel / two-finger pinch** — push the light closer to or further from the scene.
- **Panel** — intensity, ambient, relief, shadow, occlusion, bulb size, light
  color, debug views (`camera` / `depth` / `normals`), front/back camera, webcam
  resolution (`480p` / `720p` / `1080p`), depth-inference resolution
  (`252` / `392` / `448` / `518`), a **lock to depth** toggle (render only when a new
  depth map or user input arrives instead of every display refresh, so the GPU time a
  120 Hz relight would burn goes to the depth network), a live depth-inference ms/fps
  readout, and a render readout: presented fps · GPU ms of the relight pass, via timestamp
  queries. The GPU figure grows when the GPU is lightly loaded and clocks down, so expect
  a higher number at vsync than under full load.

## Credits

- Original example and relighting design: the [TypeGPU](https://typegpu.com) team
  (Software Mansion) — `apps/typegpu-docs/src/examples/image-processing/monocular-light-injection`.
- Demo photo: from the original example's assets.
- Depth model: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (Apache-2.0),
  ONNX export by the [onnx-community](https://huggingface.co/onnx-community).

## Disclaimer

This code was vibecoded — written by Claude Fable 5 (Anthropic) via
[Claude Code](https://claude.com/claude-code).
