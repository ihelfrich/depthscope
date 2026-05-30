/* DepthScope — monocular depth estimation + 3D displacement, fully client-side.
 * Depth Anything V2 (via Transformers.js, WebGPU) infers a depth map from a
 * single photo; three.js renders it as an orbitable relief surface.
 * The image never leaves the browser: there is no server doing inference. */

import { pipeline, env } from "@huggingface/transformers";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// Models load from the HF CDN and cache in the browser; never look for local copies.
env.allowLocalModels = false;

/* ---------- DOM ---------- */
const $ = (s) => document.querySelector(s);
const drop = $("#drop");
const fileInput = $("#fileInput");
const stage = $("#stage");
const loader = $("#loader");
const loaderText = $("#loaderText");
const loaderBar = $("#loaderBar");
const statusEl = $("#status");
const setStatus = (t) => (statusEl.textContent = t || "");
const setEngine = (t) => ($("#engineStatus").textContent = t);

/* ---------- tabs ---------- */
document.querySelectorAll(".tab").forEach((btn) =>
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("#" + btn.dataset.tab).classList.add("active");
    if (renderer) onResize();
  })
);

/* ---------- input wiring ---------- */
["dragenter", "dragover"].forEach((e) =>
  drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("drag"); })
);
["dragleave", "drop"].forEach((e) =>
  drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("drag"); })
);
drop.addEventListener("drop", (ev) => {
  const f = ev.dataTransfer.files && ev.dataTransfer.files[0];
  if (f) loadImageFile(f);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) loadImageFile(fileInput.files[0]);
});
document.querySelectorAll(".sample-btn").forEach((b) =>
  b.addEventListener("click", () => loadImageURL(b.dataset.src))
);
$("#resetBtn").addEventListener("click", () => {
  stage.classList.add("hidden");
  drop.classList.remove("hidden");
  fileInput.value = "";
  stopRecording();
});

/* ---------- model (lazy, once) ---------- */
let estimator = null;
const haveWebGPU = !!navigator.gpu;

async function getEstimator() {
  if (estimator) return estimator;
  loader.classList.remove("hidden");
  loaderText.textContent = haveWebGPU
    ? "Downloading depth model (first time only)…"
    : "No WebGPU here — running the slower CPU path. Downloading model…";
  estimator = await pipeline("depth-estimation", "onnx-community/depth-anything-v2-small", {
    device: haveWebGPU ? "webgpu" : "wasm",
    progress_callback: (p) => {
      if (p.status === "progress" && p.total) {
        const pct = Math.round((p.loaded / p.total) * 100);
        loaderBar.style.width = pct + "%";
        loaderText.textContent = `Downloading depth model… ${pct}%`;
      } else if (p.status === "ready") {
        loaderText.textContent = "Model ready.";
      }
    },
  });
  return estimator;
}

/* ---------- image intake ---------- */
async function loadImageFile(file) {
  if (!/^image\//.test(file.type)) { setStatus("That doesn't look like an image."); return; }
  const url = URL.createObjectURL(file);
  await runDepth(url, () => URL.revokeObjectURL(url));
}
async function loadImageURL(url) {
  await runDepth(url, null);
}

let lastDepthCanvas = null;

async function runDepth(url, cleanup) {
  drop.classList.add("hidden");
  stage.classList.remove("hidden");
  loader.classList.remove("hidden");
  loaderBar.style.width = "0%";
  setStatus("");
  onResize();

  // load the color image
  const img = new Image();
  img.crossOrigin = "anonymous";
  const colorReady = new Promise((res, rej) => {
    img.onload = () => res();
    img.onerror = () => rej(new Error("Could not load that image (a remote sample may block cross-origin use)."));
  });
  img.src = url;

  try {
    const model = await getEstimator();
    await colorReady;

    loaderText.textContent = "Estimating depth…";
    const t0 = performance.now();
    const out = await model(url);
    const ms = Math.round(performance.now() - t0);

    const depthCanvas = depthToCanvas(out.depth);
    lastDepthCanvas = depthCanvas;
    buildScene(img, depthCanvas);

    loader.classList.add("hidden");
    $("#dlDepth").disabled = false;
    $("#recBtn").disabled = false;
    setStatus(`${out.depth.width}×${out.depth.height} depth · ${haveWebGPU ? "WebGPU" : "CPU"} · ${ms} ms · drag to orbit`);
    setEngine("ready");
  } catch (err) {
    console.error(err);
    loader.classList.add("hidden");
    setStatus("⚠ " + (err.message || err));
  } finally {
    if (cleanup) cleanup();
  }
}

// RawImage -> grayscale canvas (channel-agnostic). Bright = near.
function depthToCanvas(raw) {
  const { data, width, height } = raw;
  const ch = raw.channels || Math.round(data.length / (width * height));
  const c = document.createElement("canvas");
  c.width = width; c.height = height;
  const ctx = c.getContext("2d");
  const id = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const v = data[i * ch];
    id.data[i * 4] = v; id.data[i * 4 + 1] = v; id.data[i * 4 + 2] = v; id.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
  return c;
}

/* ---------- three.js scene ---------- */
let renderer, scene, camera, controls, mesh, material, group;
let autoSpin = true, dragging = false, recording = false, recordT = 0;
let raf = null;

function initRenderer() {
  if (renderer) return;
  const canvas = $("#gl");
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, 16 / 10, 0.1, 100);
  camera.position.set(0, 0, 3.2);
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.minDistance = 1.6;
  controls.maxDistance = 6;
  controls.addEventListener("start", () => (dragging = true));
  controls.addEventListener("end", () => (dragging = false));
  scene.add(new THREE.AmbientLight(0xffffff, 1.7));
  const dir = new THREE.DirectionalLight(0xffffff, 0.55);
  dir.position.set(0.5, 0.8, 1.2);
  scene.add(dir);
  group = new THREE.Group();
  scene.add(group);
  new ResizeObserver(onResize).observe($("#viewport"));
  animate();
}

function buildScene(img, depthCanvas) {
  initRenderer();
  if (mesh) {
    group.remove(mesh);
    mesh.geometry.dispose();
    material.map?.dispose();
    material.displacementMap?.dispose();
    material.dispose();
  }
  const aspect = img.naturalWidth / img.naturalHeight;
  const W = 2.6, H = W / aspect;
  const seg = 240;
  const segX = aspect >= 1 ? seg : Math.round(seg * aspect);
  const segY = aspect >= 1 ? Math.round(seg / aspect) : seg;
  const geo = new THREE.PlaneGeometry(W, H, segX, segY);

  const colorTex = new THREE.Texture(img);
  colorTex.colorSpace = THREE.SRGBColorSpace;
  colorTex.needsUpdate = true;
  const depthTex = new THREE.CanvasTexture(depthCanvas);

  material = new THREE.MeshStandardMaterial({
    map: colorTex,
    displacementMap: depthTex,
    displacementScale: parseFloat($("#depthScale").value),
    roughness: 0.95,
    metalness: 0.0,
    side: THREE.DoubleSide,
  });
  mesh = new THREE.Mesh(geo, material);
  group.add(mesh);
  group.rotation.set(0, 0, 0);
  controls.target.set(0, 0, 0);
  camera.position.set(0, 0, 3.2);
  controls.update();
}

function onResize() {
  if (!renderer) return;
  const vp = $("#viewport");
  const w = vp.clientWidth, h = vp.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function animate() {
  raf = requestAnimationFrame(animate);
  const t = performance.now() / 1000;
  if (group) {
    if (recording) {
      recordT += 1 / 60;
      group.rotation.y = Math.sin((recordT / RECORD_SECS) * Math.PI * 2) * 0.55;
      group.rotation.x = Math.sin((recordT / RECORD_SECS) * Math.PI * 2) * 0.06;
    } else if (autoSpin && !dragging) {
      group.rotation.y = Math.sin(t * 0.6) * 0.28;
      group.rotation.x = Math.sin(t * 0.45) * 0.06;
    }
  }
  controls?.update();
  renderer?.render(scene, camera);
}

/* ---------- panel controls ---------- */
$("#depthScale").addEventListener("input", (e) => {
  if (material) material.displacementScale = parseFloat(e.target.value);
});
$("#autoSpin").addEventListener("change", (e) => {
  autoSpin = e.target.checked;
  if (!autoSpin && group) { group.rotation.set(0, 0, 0); }
});
$("#showWire").addEventListener("change", (e) => {
  if (material) material.wireframe = e.target.checked;
});
$("#dlDepth").addEventListener("click", () => {
  if (!lastDepthCanvas) return;
  lastDepthCanvas.toBlob((b) => downloadBlob(b, "depthscope_depthmap.png"));
});

/* ---------- record a clean orbit clip ---------- */
const RECORD_SECS = 6;
let recorder = null, recChunks = [];
$("#recBtn").addEventListener("click", () => (recording ? stopRecording() : startRecording()));

function startRecording() {
  if (!renderer) return;
  const btn = $("#recBtn");
  const stream = renderer.domElement.captureStream(30);
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
    ? "video/webm;codecs=vp9" : "video/webm";
  recChunks = [];
  recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8_000_000 });
  recorder.ondataavailable = (e) => e.data.size && recChunks.push(e.data);
  recorder.onstop = () => {
    const blob = new Blob(recChunks, { type: "video/webm" });
    downloadBlob(blob, "depthscope_orbit.webm");
    setStatus("Saved orbit clip.");
  };
  recording = true; recordT = 0;
  btn.textContent = "Recording…"; btn.classList.add("rec-on");
  recorder.start();
  setTimeout(() => { if (recording) stopRecording(); }, RECORD_SECS * 1000 + 120);
}
function stopRecording() {
  if (!recording) return;
  recording = false;
  const btn = $("#recBtn");
  btn.textContent = "Record orbit (.webm)"; btn.classList.remove("rec-on");
  if (group) group.rotation.set(0, 0, 0);
  try { recorder && recorder.state !== "inactive" && recorder.stop(); } catch (_) {}
}

/* ---------- util ---------- */
function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

setEngine(haveWebGPU ? "WebGPU ready" : "CPU fallback");
