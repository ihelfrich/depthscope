# DepthScope

> **Moved.** DepthScope is now the Depth tool in [Scope](https://github.com/ihelfrich/scope) and lives at https://ihelfrich.github.io/scope/depth/. This repository is kept so old links redirect; development continues in `scope/depth/`.

Drop a flat photo and watch a neural network infer its depth and pop it into an interactive 3D scene you can orbit, light, and record. The model runs on your own GPU through WebGPU. Your photo is never uploaded, because there is no server doing the work.

**Live:** https://ihelfrich.github.io/depthscope

Part of the `*-Scope` family: [DocScope](https://ihelfrich.github.io/docscope) (documents to text), [MetaScope](https://ihelfrich.github.io/metascope) (what your files reveal), DepthScope (photos to 3D).

## What it does

A single photograph carries no depth information. It is a flat grid of colors. A monocular depth model has learned, from millions of examples, to guess which parts are near and which are far. DepthScope runs that model in the browser, turns the guess into a relief surface, and lets you fly around it.

1. **Estimate.** Depth Anything V2 (small) takes your photo and outputs a depth map: bright is near, dark is far.
2. **Displace.** A fine three.js mesh is textured with your photo and pushed toward or away from the camera by the depth map.
3. **Play.** Orbit with the mouse, exaggerate or calm the effect with the depth-pop slider, toggle a wireframe to see the geometry, save the depth map as a PNG, or record a clean orbit as a `.webm` clip to share.

## Why this is surprising

The inference happens on your machine. The first photo is slower because the model file (~100 MB) downloads once and is then cached by the browser; after that, estimation takes a second or two on a typical GPU. No image, no depth map, no pixel is sent anywhere. A few years ago this needed a server with a dedicated GPU. Now it fits in a tab.

## Best inputs

Photos with a clear foreground and a receding background pop the most: portraits, a person on a street, a path into a landscape. Flat head-on shots (a document, a wall) have little real depth for the model to find. The depth-pop slider exaggerates or softens the displacement.

## Stack

All client-side, loaded from CDN as ES modules. No build step.

- [Transformers.js](https://github.com/huggingface/transformers.js) — runs Depth Anything V2 (ONNX) on WebGPU, with a slower CPU (WASM) fallback
- [three.js](https://threejs.org) — the displacement mesh, orbit controls, and render
- `MediaRecorder` + `captureStream` — the shareable orbit clip

```
depthscope/
├── index.html        UI + import map
├── css/style.css
├── js/app.js         model load + depth + 3D render (ES module)
├── .nojekyll
└── README.md
```

## Honest limits

- **WebGPU required for the fast path.** Coverage is broad on current desktops and recent phones, but some browsers (Linux Firefox, older mobile) lack it and fall back to a much slower CPU mode, or are unsupported. The page detects this and tells you.
- **First load downloads ~100 MB** (the fp32 model). It is cached afterward. A future option could ship an fp16 build to halve this.
- **Monocular depth is a guess, not a measurement.** It infers relative, not metric, depth and can be fooled by reflections, glass, and unusual scenes. This is a toy that happens to use a serious model, not a metrology tool.

## License

MIT. Model weights are licensed by their authors (Depth Anything V2, Apache-2.0).
