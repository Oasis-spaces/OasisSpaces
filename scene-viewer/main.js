// The mixed-scene viewer: the room's shell is a textured mesh (shell.glb), each
// piece of furniture is its own splat (pieces/*.splat), both drawn together by
// three.js and Spark. Pieces are selected by their measured boxes, dragged
// across the floor, turned about the vertical, hidden and put back. Everything
// is in metres, y up, the room's centre on the floor at the origin (see
// tools/mixed_scene.py, which writes scene.json).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

const params = new URLSearchParams(location.search);
const sceneURL = new URL(params.get("scene") || "../spaces/walkthrough-full/scene/scene.json", location.href);
const status = document.getElementById("status");

const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101012);
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.05, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.495;
controls.minDistance = 0.3;
controls.maxDistance = 14;

scene.add(new SparkRenderer({ renderer }));
// The shell's textures already hold the room's own light, so the shell is drawn as it is
// (unlit). The lights are for the clean models: soft light from above, a key light for
// shape, and a room environment for the sheen of wood and metal.
scene.add(new THREE.HemisphereLight(0xffffff, 0x8d8a80, 1.9));
const key = new THREE.DirectionalLight(0xfff4e6, 1.3);
key.position.set(2.5, 5, 3);
scene.add(key);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.35;

const pieces = new Map();   // id -> {info, group, proxy, row, home: {position, rotation}}
let manifest = null;
let selected = null;
const outline = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xff9f0a }));
outline.visible = false;
outline.renderOrder = 10;
outline.material.depthTest = false;
scene.add(outline);

async function load() {
  manifest = await (await fetch(sceneURL)).json();
  document.getElementById("title").textContent = manifest.space.replace(/[-_]/g, " ");
  document.getElementById("subtitle").textContent =
    `${manifest.room.width.toFixed(2)} × ${manifest.room.depth.toFixed(2)} m, ${manifest.room.height.toFixed(2)} m high`;
  document.title = `${manifest.space} · Oasis scene`;
  if (manifest.summary) document.getElementById("summary").textContent = manifest.summary;

  const shell = await new GLTFLoader().loadAsync(new URL(manifest.shell, sceneURL).href);
  shell.scene.traverse((node) => {
    if (node.isMesh) {
      const map = node.material.map;
      map.anisotropy = renderer.capabilities.getMaxAnisotropy();
      node.material = new THREE.MeshBasicMaterial({ map, side: THREE.FrontSide });
      node.userData.shell = true;
    }
  });
  scene.add(shell.scene);

  for (const info of manifest.pieces) addPiece(info);
  // A thing standing on another moves with it.
  for (const piece of pieces.values()) {
    const parent = piece.info.on && pieces.get(piece.info.on);
    if (parent) parent.group.attach(piece.group);
  }
  for (const piece of pieces.values()) {
    piece.home = { position: piece.group.position.clone(), rotation: piece.group.rotation.y };
  }
  viewFromAbove(false);
  status.classList.add("done");
}

function addPiece(info) {
  const group = new THREE.Group();
  group.position.fromArray(info.anchor);
  const splat = new SplatMesh({ url: new URL(info.file, sceneURL).href });
  group.add(splat);
  // The clean stand-in, beside the scan: one of the two shows.
  const model = new THREE.Group();
  if (info.model) {
    new GLTFLoader().loadAsync(new URL(info.model, sceneURL).href).then((gltf) => model.add(gltf.scene));
    group.add(model);
  }
  const showing = info.show === "model" && info.model ? "model" : "scan";
  splat.visible = showing === "scan";
  model.visible = showing === "model";
  const min = new THREE.Vector3().fromArray(info.box.min), max = new THREE.Vector3().fromArray(info.box.max);
  const size = max.clone().sub(min);
  let proxy = null;
  if (info.movable) {
    proxy = new THREE.Mesh(new THREE.BoxGeometry(size.x, size.y, size.z),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, colorWrite: false }));
    proxy.position.copy(min).add(max).multiplyScalar(0.5);
    proxy.userData.piece = info.id;
    group.add(proxy);
  }
  scene.add(group);

  const row = document.createElement("label");
  row.className = "row";
  row.innerHTML = `<input type="checkbox" checked><span class="name"></span><span class="count"></span>`;
  row.querySelector(".name").textContent = info.label;
  row.querySelector(".count").textContent = info.movable ? (showing === "model" ? "model" : "scan") : "";
  const box = row.querySelector("input");
  box.addEventListener("change", () => { group.visible = box.checked; if (!box.checked && selected === info.id) select(null); });
  row.addEventListener("click", (event) => { if (event.target !== box && info.movable && group.visible) { event.preventDefault(); select(info.id); } });
  document.getElementById(info.movable ? "furniture" : "fixed").append(row);
  pieces.set(info.id, { info, group, proxy, row, size, splat, model, showing });
}

// ---------------------------------------------------------------- selection
function select(id) {
  selected = id;
  for (const piece of pieces.values()) piece.row.classList.toggle("selected", piece.info.id === id);
  const bar = document.getElementById("selection");
  if (!id) { outline.visible = false; bar.hidden = true; return; }
  const piece = pieces.get(id);
  document.getElementById("sel-title").textContent = piece.info.label;
  document.getElementById("sel-size").textContent =
    `${piece.size.x.toFixed(2)} × ${piece.size.z.toFixed(2)} × ${piece.size.y.toFixed(2)} m`;
  const swap = document.getElementById("swap");
  swap.hidden = !piece.info.model;
  swap.textContent = piece.showing === "scan" ? "Show clean model" : "Show the scan";
  swap.title = piece.info.why || "";
  bar.hidden = false;
  outline.geometry.dispose();
  outline.geometry = new THREE.EdgesGeometry(piece.proxy.geometry);
  outline.visible = true;
}

function followOutline() {
  if (!selected) return;
  const proxy = pieces.get(selected).proxy;
  proxy.updateWorldMatrix(true, false);
  outline.matrixAutoUpdate = false;
  outline.matrix.copy(proxy.matrixWorld);
  outline.matrixWorld.copy(proxy.matrixWorld);
}

const ray = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const floor = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let drag = null;

function aim(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
  ray.setFromCamera(pointer, camera);
}

function pieceUnder(event) {
  aim(event);
  const proxies = [...pieces.values()].filter((p) => p.proxy && p.group.visible).map((p) => p.proxy);
  const hit = ray.intersectObjects(proxies, false)[0];
  return hit ? hit.object.userData.piece : null;
}

renderer.domElement.addEventListener("pointerdown", (event) => {
  const id = pieceUnder(event);
  const downAt = [event.clientX, event.clientY];
  if (id && id === selected) {
    const piece = pieces.get(id);
    const hit = new THREE.Vector3();
    if (ray.ray.intersectPlane(floor, hit)) {
      const world = piece.group.getWorldPosition(new THREE.Vector3());
      drag = { id, offset: world.sub(hit) };
      controls.enabled = false;
      renderer.domElement.setPointerCapture(event.pointerId);
    }
  }
  const up = (e) => {
    renderer.domElement.removeEventListener("pointerup", up);
    const moved = Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5;
    if (!drag && !moved) select(id);          // a click, not an orbit
    drag = null;
    controls.enabled = true;
  };
  renderer.domElement.addEventListener("pointerup", up);
});

renderer.domElement.addEventListener("pointermove", (event) => {
  if (!drag) { renderer.domElement.style.cursor = pieceUnder(event) ? "pointer" : "default"; return; }
  aim(event);
  const hit = new THREE.Vector3();
  if (!ray.ray.intersectPlane(floor, hit)) return;
  const piece = pieces.get(drag.id);
  const target = hit.add(drag.offset);
  // Keep the piece's centre inside the room.
  const halfW = manifest.room.width / 2, halfD = manifest.room.depth / 2;
  target.x = THREE.MathUtils.clamp(target.x, -halfW, halfW);
  target.z = THREE.MathUtils.clamp(target.z, -halfD, halfD);
  target.y = piece.group.getWorldPosition(new THREE.Vector3()).y;
  piece.group.parent.worldToLocal(target);
  piece.group.position.copy(target);
});

function showAs(id, what) {
  const piece = pieces.get(id);
  if (!piece || !piece.info.model) return;
  piece.showing = what;
  piece.splat.visible = what === "scan";
  piece.model.visible = what === "model";
  piece.row.querySelector(".count").textContent = what;
  if (selected === id) document.getElementById("swap").textContent = what === "scan" ? "Show clean model" : "Show the scan";
}

function turn(degrees) {
  if (selected) pieces.get(selected).group.rotation.y += THREE.MathUtils.degToRad(degrees);
}
function hideSelected() {
  if (!selected) return;
  const piece = pieces.get(selected);
  piece.group.visible = false;
  piece.row.querySelector("input").checked = false;
  select(null);
}
function putBack(id) {
  const piece = pieces.get(id);
  piece.group.position.copy(piece.home.position);
  piece.group.rotation.y = piece.home.rotation;
}

document.getElementById("turn-left").onclick = () => turn(15);
document.getElementById("turn-right").onclick = () => turn(-15);
document.getElementById("hide").onclick = hideSelected;
document.getElementById("swap").onclick = () => selected && showAs(selected, pieces.get(selected).showing === "scan" ? "model" : "scan");
document.getElementById("all-models").onclick = () => { for (const p of pieces.values()) showAs(p.info.id, "model"); };
document.getElementById("all-scans").onclick = () => { for (const p of pieces.values()) showAs(p.info.id, "scan"); };
document.getElementById("menu").onclick = () => document.getElementById("panel").classList.toggle("open");
document.getElementById("put-back").onclick = () => selected && putBack(selected);
document.getElementById("reset").onclick = () => {
  for (const piece of pieces.values()) {
    putBack(piece.info.id);
    piece.group.visible = true;
    piece.row.querySelector("input").checked = true;
  }
};
document.getElementById("export").onclick = () => {
  const layout = { space: manifest.space, pieces: [...pieces.values()].filter((p) => p.info.movable).map((p) => {
    const position = p.group.getWorldPosition(new THREE.Vector3());
    return { id: p.info.id, label: p.info.label, hidden: !p.group.visible, show: p.showing,
             position: position.toArray().map((v) => +v.toFixed(3)),
             turnDegrees: +THREE.MathUtils.radToDeg(p.group.rotation.y).toFixed(1) };
  }) };
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([JSON.stringify(layout, null, 1)], { type: "application/json" }));
  link.download = `${manifest.space}-layout.json`;
  link.click();
};
window.addEventListener("keydown", (event) => {
  if (event.key === "q" || event.key === "Q") turn(15);
  if (event.key === "e" || event.key === "E") turn(-15);
  if (event.key === "h" || event.key === "H" || event.key === "Delete" || event.key === "Backspace") hideSelected();
  if (event.key === "m" || event.key === "M") selected && showAs(selected, pieces.get(selected).showing === "scan" ? "model" : "scan");
  if (event.key === "Escape") select(null);
});

// -------------------------------------------------------------------- views
function moveCamera(position, target, animate = true) {
  const from = camera.position.clone(), fromTarget = controls.target.clone();
  const to = new THREE.Vector3(...position), toTarget = new THREE.Vector3(...target);
  if (!animate) { camera.position.copy(to); controls.target.copy(toTarget); return; }
  const start = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - start) / 700), k = t * t * (3 - 2 * t);
    camera.position.lerpVectors(from, to, k);
    controls.target.lerpVectors(fromTarget, toTarget, k);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
function show(id, visible) {
  const piece = pieces.get(id);
  if (!piece) return;
  piece.group.visible = visible;
  piece.row.querySelector("input").checked = visible;
}
function viewFromAbove(animate = true) {
  const r = manifest.room, reach = Math.max(r.width, r.depth);
  controls.maxPolarAngle = Math.PI * 0.495;
  show("ceiling-fittings", false);           // a fan and its haze would hide the room from above
  moveCamera([reach * 0.55, r.height + reach * 0.9, reach * 1.05], [0, r.height * 0.3, 0], animate);
}
function viewFromInside() {
  const r = manifest.room;
  controls.maxPolarAngle = Math.PI * 0.95;
  show("ceiling-fittings", true);
  // From where the person stood while filming, looking at the middle of the room. The orbit
  // target sits just ahead of the eye, so dragging looks around instead of circling the room.
  const eye = new THREE.Vector3(...(manifest.inside?.position ?? [0, 1.5, r.depth * 0.3]));
  const towards = new THREE.Vector3(...(manifest.inside?.target ?? [0, 1.1, 0])).sub(eye).setLength(0.4);
  moveCamera(eye.toArray(), eye.clone().add(towards).toArray());
}
document.getElementById("view-top").onclick = () => viewFromAbove();
document.getElementById("view-inside").onclick = viewFromInside;

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

renderer.setAnimationLoop(() => {
  controls.update();
  followOutline();
  renderer.render(scene, camera);
});

window.oasisScene = { pieces, scene, camera };   // for checks from outside (the apps, tests)
load().catch((error) => { status.textContent = `Could not load the scene: ${error.message}`; console.error(error); });
