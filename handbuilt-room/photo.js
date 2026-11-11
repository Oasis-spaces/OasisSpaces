// The hand-built bedroom with photographs on it: room.json gives the shell and the
// big pieces, textures/ holds one image per surface, baked from the video's frames
// by bake_textures.py. A surface the video barely saw keeps a plain colour.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const params = new URLSearchParams(location.search);
if (params.has("shot")) document.body.classList.add("shot");
const MIN_SEEN = 0.45;                      // below this share of a surface seen, the photo is not used
const PLAIN = { floor: "#cfc8b6", ceiling: "#efefec" };

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x15171a);
const camera = new THREE.PerspectiveCamera(69, innerWidth / innerHeight, 0.05, 60);
const controls = new OrbitControls(camera, renderer.domElement);

const [room, coverage] = await Promise.all([
  fetch("./room.json").then((r) => r.json()),
  fetch("./textures/coverage.json").then((r) => r.json()).catch(() => ({})),
]);
const { W, L, H } = room.size;
// Plan coordinates are the mirror image of three's, so the group is flipped once.
const group = new THREE.Group(); group.scale.x = -1; group.position.x = W; scene.add(group);

const bodies = {};
for (const b of room.boxes) {
  if (b.draw === false) continue;
  const size = b.max.map((v, i) => v - b.min[i]);
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size),
    new THREE.MeshBasicMaterial({ color: b.color || "#555", polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 2 }));
  mesh.position.set(...b.min.map((v, i) => v + size[i] / 2));
  group.add(mesh); bodies[b.name] = mesh;
}

const loader = new THREE.TextureLoader();
let ceiling = null;
for (const s of room.surfaces) {
  const O = new THREE.Vector3(...s.origin), U = new THREE.Vector3(...s.u).multiplyScalar(s.w), V = new THREE.Vector3(...s.v).multiplyScalar(s.h);
  const corners = [O, O.clone().add(U), O.clone().add(U).add(V), O.clone().add(V)];
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(corners.flatMap((p) => [p.x, p.y, p.z]), 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
  g.setIndex([0, 1, 2, 0, 2, 3]);
  const seen = coverage[s.name] ?? 0;
  const shell = !s.of;                      // walls, floor, ceiling: one-sided, so the room can be looked into
  let material;
  if (s.name !== "ceiling" && (seen >= MIN_SEEN || s.name === "wall-wardrobe")) {   // the ceiling was filmed in scraps: kept plain
    const map = loader.load(`./textures/${s.name}.jpg`); map.colorSpace = THREE.SRGBColorSpace; map.anisotropy = 8;
    material = new THREE.MeshBasicMaterial({ map, side: shell ? THREE.FrontSide : THREE.DoubleSide });
  } else if (shell) {
    material = new THREE.MeshBasicMaterial({ color: PLAIN[s.name] || "#e9e8e3" });
  } else continue;                          // an unseen face of a piece: its plain body shows
  const mesh = new THREE.Mesh(g, material);
  group.add(mesh);
  if (s.name === "ceiling") ceiling = mesh;
}

const VIEWS = {
  "Dollhouse": { from: [5.4, 4.6, 5.9], to: [W / 2, 0.9, L / 2], fov: 40, open: true },
  "From above": { from: [W / 2, 7.6, L / 2 + 0.01], to: [W / 2, 0, L / 2], fov: 36, open: true },
  "From the desk": { from: [2.3, 1.45, 2.9], to: [1.0, 1.1, 0], fov: 69 },
  "From the door": { from: [2.4, 1.4, 0.35], to: [1.6, 1.0, L], fov: 69 },
  "Across the bed": { from: [2.3, 1.45, 1.0], to: [0, 1.15, 2.2], fov: 69 },
  "Bedside": { from: [1.5, 1.35, 2.6], to: [0, 1.25, 1.2], fov: 69 },
};
function setView(name) {
  const v = VIEWS[name] || VIEWS["Dollhouse"];
  camera.fov = v.fov; camera.position.set(W - v.from[0], v.from[1], v.from[2]);
  controls.target.set(W - v.to[0], v.to[1], v.to[2]);
  if (ceiling) ceiling.visible = !v.open;
  camera.updateProjectionMatrix(); controls.update();
}
const bar = document.getElementById("bar");
for (const name of Object.keys(VIEWS)) { const b = document.createElement("button"); b.textContent = name;
  b.onclick = () => setView(name); bar.appendChild(b); }
setView(params.get("view") || "Dollhouse");

addEventListener("resize", () => { renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); });
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
window.roomReady = true;
