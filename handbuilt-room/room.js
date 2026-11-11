// The bedroom from videos/IMG_4182.MOV, modelled by hand from its frames: no
// camera solve, no depth, no detector. Sizes are estimated from things of known
// size (door 0.8 x 2.05 m, double bed 1.8 x 2.0 m, wardrobe 0.6 m deep).
//
// Plan (metres). x runs from the headboard wall (H, x = 0) to the desk wall
// (D, x = W). z runs from the wardrobe wall (S, z = 0) to the curtain wall
// (C, z = L). y is up.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const W = 3.1, L = 3.6, H = 2.75;
const params = new URLSearchParams(location.search);
if (params.has("shot")) document.body.classList.add("shot");

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x15171a);
const camera = new THREE.PerspectiveCamera(69, innerWidth / innerHeight, 0.05, 60);
const controls = new OrbitControls(camera, renderer.domElement);

// ---------- textures, drawn ----------
function rng(seed) { let s = seed; return () => (s = (s * 16807) % 2147483647) / 2147483647; }
function canvasTexture(size, draw, repeat) {
  const c = document.createElement("canvas"); c.width = c.height = size;
  draw(c.getContext("2d"), size);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping; if (repeat) t.repeat.set(...repeat);
  t.anisotropy = 8; return t;
}
// The grey petal laminate on the wardrobe panels, the shelf and the desk top.
const petals = (repeat) => canvasTexture(512, (g, s) => {
  const r = rng(7); g.fillStyle = "#cfd6dc"; g.fillRect(0, 0, s, s);
  const tones = ["#8f97a3", "#aab3be", "#e6ebef", "#77808c", "#b9b3c6", "#d9dcc6", "#9fb4c4", "#5f6672"];
  for (let i = 0; i < 260; i++) {
    g.save(); g.translate(r() * s, r() * s); g.rotate(r() * Math.PI * 2);
    const w = 22 + r() * 46; g.globalAlpha = 0.5 + r() * 0.4; g.fillStyle = tones[Math.floor(r() * tones.length)];
    g.beginPath(); g.moveTo(0, 0); g.quadraticCurveTo(w * 0.7, -w * 0.5, w, 0); g.quadraticCurveTo(w * 0.7, w * 0.5, 0, 0); g.fill();
    g.restore();
  }
}, repeat);
const stucco = canvasTexture(256, (g, s) => {
  const r = rng(3); g.fillStyle = "#eeeeea"; g.fillRect(0, 0, s, s);
  for (let i = 0; i < 2600; i++) { g.fillStyle = r() > 0.5 ? "rgba(255,255,255,.7)" : "rgba(170,170,165,.35)";
    g.fillRect(r() * s, r() * s, 1 + r() * 2, 3 + r() * 9); }
}, [5, 4]);
const tiles = canvasTexture(256, (g, s) => {
  g.fillStyle = "#d8d2c2"; g.fillRect(0, 0, s, s); g.strokeStyle = "#bdb6a4"; g.lineWidth = 3; g.strokeRect(0, 0, s, s);
}, [W / 0.6, L / 0.6]);
const slats = canvasTexture(256, (g, s) => {
  const r = rng(11); g.fillStyle = "#6b4630"; g.fillRect(0, 0, s, s);
  for (let i = 0; i < 400; i++) { g.fillStyle = `rgba(${40 + r() * 60},${20 + r() * 30},10,.18)`; g.fillRect(0, r() * s, s, 1); }
  g.fillStyle = "#3d2617"; for (let y = 0; y < s; y += s / 6) g.fillRect(0, y, s, 3);
});
const sheet = canvasTexture(512, (g, s) => {
  const r = rng(5); g.fillStyle = "#8d8a86"; g.fillRect(0, 0, s, s);
  for (let i = 0; i < 16; i++) {
    const x = r() * s, y = r() * s, rad = 28 + r() * 38, kind = r();
    g.beginPath(); g.arc(x, y, rad, 0, 7);
    g.fillStyle = kind < 0.4 ? "#25252a" : kind < 0.7 ? "#e9e6dc" : "#a9b89a"; g.fill();
    if (kind < 0.4) { g.fillStyle = "#d9d6cf"; for (let k = 0; k < 40; k++) { const a = r() * 7, d = r() * rad * 0.9;
      g.fillRect(x + Math.cos(a) * d, y + Math.sin(a) * d, 3, 3); } }
  }
}, [2, 2]);
const leaves = canvasTexture(256, (g, s) => {
  const r = rng(9); g.fillStyle = "#efe8da"; g.fillRect(0, 0, s, s);
  const tones = ["#d9703c", "#9a9a3a", "#2b2b2b", "#e7dfcc", "#c9503a"];
  for (let i = 0; i < 14; i++) { g.save(); g.translate(r() * s, r() * s); g.rotate(r() * 6.3);
    const w = 50 + r() * 40; g.fillStyle = tones[Math.floor(r() * tones.length)]; g.strokeStyle = "#333"; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(0, 0); g.quadraticCurveTo(w * 0.5, -w * 0.45, w, 0); g.quadraticCurveTo(w * 0.5, w * 0.45, 0, 0);
    g.fill(); g.stroke(); g.beginPath(); g.moveTo(0, 0); g.lineTo(w, 0); g.stroke(); g.restore(); }
});
const folds = (base, dark) => canvasTexture(256, (g, s) => {
  for (let x = 0; x < s; x++) { const t = 0.5 + 0.5 * Math.sin(x / s * Math.PI * 2 * 7);
    g.fillStyle = t > 0.5 ? base : dark; g.globalAlpha = 1; g.fillRect(x, 0, 1, s);
    g.globalAlpha = 0.25 * Math.abs(t - 0.5) * 2; g.fillStyle = t > 0.5 ? "#fff" : "#000"; g.fillRect(x, 0, 1, s); }
  g.globalAlpha = 0.12; g.fillStyle = "#000"; for (let y = 0; y < s; y += 16) g.fillRect(0, y, s, 1);
});
const weave = canvasTexture(64, (g, s) => { g.fillStyle = "#7d7a5c"; g.fillRect(0, 0, s, s);
  g.fillStyle = "rgba(40,40,20,.25)"; for (let i = 0; i < s; i += 4) { g.fillRect(i, 0, 1, s); g.fillRect(0, i, s, 1); } }, [6, 6]);

// ---------- materials ----------
const mat = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.85, ...extra });
const M = {
  wall: mat(0xecebe6), stucco: mat(0xffffff, { map: stucco }), floor: mat(0xffffff, { map: tiles, roughness: 0.6 }),
  ceiling: mat(0xf4f4f1), plum: mat(0x4a3f47), petals: mat(0xffffff, { map: petals([1, 1]) }),
  door: mat(0xd8cfb4, { roughness: 0.6 }), frame: mat(0xc9c0a4), wood: mat(0xffffff, { map: slats }), woodPlain: mat(0x5d3b27),
  sheet: mat(0xffffff, { map: sheet }), leaves: mat(0xffffff, { map: leaves }), olive: mat(0xffffff, { map: weave }),
  green: mat(0xffffff, { map: folds("#bccf73", "#a3b85c") }), blue: mat(0xffffff, { map: folds("#5f9bb8", "#4c86a3") }),
  steel: mat(0xb9bcc0, { roughness: 0.35, metalness: 0.7 }), maroon: mat(0x7a2530), black: mat(0x141416, { roughness: 0.4 }),
  white: mat(0xf6f6f2), ac: mat(0xf1ecd9), pink: mat(0xb9597c), navy: mat(0x1f2a44), grey: mat(0x8a8d90),
  mirror: mat(0xdfe8ea, { roughness: 0.08, metalness: 0.6 }), red: mat(0xb3352f), paper: mat(0xf2a7c3), yellow: mat(0xf2d96b),
  glow: new THREE.MeshBasicMaterial({ color: 0xffffff }),
};

// ---------- helpers ----------
// The plan is drawn with x to the right and z away; seen from inside, that is the
// mirror image of the real room, so the whole group is flipped once here.
const room = new THREE.Group(); room.scale.x = -1; room.position.x = W; scene.add(room);
// A box given by its lowest corner and its size.
function box(x, y, z, w, h, d, material, parent = room) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
  m.position.set(x + w / 2, y + h / 2, z + d / 2); m.castShadow = m.receiveShadow = true; parent.add(m); return m;
}
// One-sided surface, so the room can be looked into from outside.
function surface(w, h, material, position, rotation) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), material);
  m.position.set(...position); m.rotation.set(...rotation); m.receiveShadow = true; room.add(m); return m;
}

// ---------- shell ----------
surface(W, L, M.floor, [W / 2, 0, L / 2], [-Math.PI / 2, 0, 0]);
const ceiling = surface(W, L, M.ceiling, [W / 2, H, L / 2], [Math.PI / 2, 0, 0]);
surface(L, H, M.wall, [0, H / 2, L / 2], [0, Math.PI / 2, 0]);        // H: headboard wall
surface(L, H, M.wall, [W, H / 2, L / 2], [0, -Math.PI / 2, 0]);       // D: desk wall
surface(W, H, M.wall, [W / 2, H / 2, 0], [0, 0, 0]);                  // S: wardrobe wall
surface(W, H, M.stucco, [W / 2, H / 2, L], [0, Math.PI, 0]);          // C: curtain wall (textured paint)

function doorLeaf(x, z, facing, handleRight) {   // facing: +1 looks along +z, -1 along -z
  const zf = facing > 0 ? z : z - 0.04;
  box(x - 0.06, 0, zf, 0.92, 2.11, 0.04, M.frame);
  box(x, 0, zf + facing * 0.012, 0.8, 2.05, 0.04, M.door);
  box(handleRight ? x + 0.68 : x + 0.08, 1.0, zf + facing * 0.05, 0.04, 0.16, 0.03, M.steel);
}
doorLeaf(W - 0.95, 0, +1, false);   // door in the wardrobe wall, beside the dresser
doorLeaf(0.1, L, -1, true);         // door in the curtain wall, beside the headboard corner

// ---------- wardrobe wall: dresser niche, sliding wardrobe, loft across the top ----------
const WD = 0.6, wardrobeEnd = 2.1, niche = 0.5;                  // wardrobe runs x 0 .. 2.1; niche is its last 0.5
box(0, 0, 0, wardrobeEnd - niche, 2.1, WD, M.plum);              // sliding wardrobe carcass
for (const [x0, w] of [[0.02, 0.78], [0.82, 0.76]]) {            // two sliding doors with petal panels
  box(x0, 0.05, WD, w, 2.0, 0.02, M.plum);
  if (x0 < 0.5) box(x0 + 0.12, 0.3, WD + 0.02, w - 0.24, 1.5, 0.006, M.petals);   // the door by the corner carries the panel
  box(x0 + (x0 < 0.5 ? w - 0.07 : 0.03), 1.0, WD + 0.02, 0.035, 0.14, 0.012, M.steel);
}
const nx = wardrobeEnd - niche;                                   // dresser niche
box(nx, 0, 0, niche, 0.95, WD, M.plum);
box(nx + 0.05, 0.08, WD, niche - 0.1, 0.42, 0.012, M.petals);
for (const y of [0.56, 0.76]) { box(nx + 0.04, y, WD, niche - 0.08, 0.17, 0.015, M.petals);
  box(nx + niche / 2 - 0.05, y + 0.07, WD + 0.015, 0.1, 0.025, 0.02, M.steel); }
box(nx, 0.95, 0, 0.03, 1.15, WD, M.plum); box(wardrobeEnd - 0.03, 0.95, 0, 0.03, 1.15, WD, M.plum);
box(nx + 0.03, 0.95, 0.02, niche - 0.06, 1.15, 0.02, M.mirror);
box(nx + 0.03, 1.0, 0.1, niche - 0.06, 0.3, 0.3, M.black);        // clothes heaped in the niche
// Loft cabinets over the door and the wardrobe, wall to wall.
box(0, 2.1, 0, W, H - 2.1, WD, M.plum);
for (let i = 0; i < 4; i++) { const lw = W / 4; box(i * lw + 0.04, 2.19, WD, lw - 0.08, 0.42, 0.012, M.petals);
  box(i * lw + lw / 2 - 0.015, 2.13, WD + 0.012, 0.03, 0.09, 0.02, M.steel); }

// ---------- bed ----------
const bedZ = 1.0, bedW = 1.8, bedL = 2.05;
box(0.05, 0.12, bedZ, bedL - 0.05, 0.22, bedW, M.woodPlain);                       // frame
box(bedL - 0.04, 0, bedZ, 0.04, 0.5, bedW, M.woodPlain);                           // footboard
box(0.09, 0.34, bedZ + 0.03, bedL - 0.16, 0.2, bedW - 0.06, M.sheet);              // mattress and sheet
{ // headboard: slatted, with a curved top
  const shape = new THREE.Shape(); shape.moveTo(0, 0); shape.lineTo(bedW, 0); shape.lineTo(bedW, 0.82);
  shape.quadraticCurveTo(bedW / 2, 1.12, 0, 0.82); shape.lineTo(0, 0);
  const g = new THREE.ExtrudeGeometry(shape, { depth: 0.06, bevelEnabled: false });
  const uv = g.attributes.uv; for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) / bedW, uv.getY(i) / 1.1);
  const m = new THREE.Mesh(g, M.wood); m.rotation.y = Math.PI / 2; m.position.set(0.01, 0, bedZ + bedW);
  m.castShadow = true; room.add(m);
}
function pillow(x, z, material) {              // flat sleeping pillow
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.5, 20, 12), material);
  m.scale.set(0.44, 0.15, 0.66); m.position.set(x, 0.6, z); m.rotation.z = 0.25; m.castShadow = true; room.add(m);
}
function cushion(z) {                          // square cushion leaning on the headboard
  const m = box(0, 0, 0, 0.12, 0.46, 0.6, M.olive); m.position.set(0.2, 0.78, z); m.rotation.z = -0.22;
}
cushion(bedZ + 0.47); cushion(bedZ + 1.33);
pillow(0.5, bedZ + 0.47, M.leaves); pillow(0.5, bedZ + 1.33, M.leaves);
{ // the laptop and the black backpack left on the bed
  box(1.25, 0.545, bedZ + 0.25, 0.24, 0.015, 0.34, M.grey);
  const lid = box(1.25, 0.56, bedZ + 0.25, 0.012, 0.23, 0.34, M.black); lid.rotation.z = -0.3; lid.position.x -= 0.03;
  const bag = new THREE.Mesh(new THREE.SphereGeometry(0.5, 16, 10), M.black);
  bag.scale.set(0.42, 0.2, 0.34); bag.position.set(1.5, 0.62, bedZ + 1.35); bag.castShadow = true; room.add(bag);
}

// ---------- shelf above the headboard: two frames joined by a zigzag ----------
function frame(z0, y0, w, h, d = 0.2, t = 0.045) {
  box(0, y0, z0, d, t, w, M.petals); box(0, y0 + h - t, z0, d, t, w, M.petals);
  box(0, y0, z0, d, h, t, M.petals); box(0, y0, z0 + w - t, d, h, t, M.petals);
  box(0, y0 + t, z0 + t, 0.012, h - 2 * t, w - 2 * t, M.plum);
}
frame(1.0, 1.55, 0.75, 0.75);
frame(2.0, 1.75, 0.8, 0.6);
{ const zig = box(0, 1.7, 1.72, 0.2, 0.045, 0.42, M.petals); zig.rotation.x = 0.75;
  box(0, 1.9, 1.0, 0.2, 0.04, 0.75, M.petals);                                    // inner shelf of the first frame
  box(0.03, 1.6, 1.1, 0.09, 0.09, 0.22, M.black); box(0.04, 1.6, 1.5, 0.08, 0.1, 0.08, M.white); }

// ---------- curtain wall: AC, rod, two curtains, bench, pin board ----------
box(1.3, 2.32, L - 0.22, 0.9, 0.3, 0.22, M.ac);
box(1.34, 2.33, L - 0.225, 0.82, 0.05, 0.01, M.black);
{ const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 1.6, 12), M.steel);
  rod.rotation.z = Math.PI / 2; rod.position.set(2.0, 2.12, L - 0.1); room.add(rod); }
function curtain(x0, w, material) {
  const g = new THREE.PlaneGeometry(w, 2.08, 28, 1), p = g.attributes.position;
  for (let i = 0; i < p.count; i++) p.setZ(i, 0.035 * Math.sin((p.getX(i) / w + 0.5) * Math.PI * 2 * 7));
  g.computeVertexNormals();
  const m = new THREE.Mesh(g, material); m.material.side = THREE.DoubleSide;
  m.position.set(x0 + w / 2, 1.06, L - 0.1); m.rotation.y = Math.PI; m.castShadow = true; room.add(m);
}
curtain(1.28, 0.62, M.green); curtain(1.92, 0.7, M.blue);
box(2.69, 1.45, L - 0.09, 0.02, 0.62, 0.02, M.navy);                               // tie-back tassel
box(1.05, 1.95, L - 0.012, 0.09, 0.09, 0.012, M.white);                            // socket
box(2.66, 0.78, L - 0.015, W - 2.66, 0.72, 0.015, M.maroon);                       // pin board
box(2.74, 0.86, L - 0.02, 0.2, 0.3, 0.006, M.black);                               // poster
for (const [x, y, c] of [[2.69, 1.28, M.paper], [2.98, 1.34, M.yellow], [2.72, 1.2, M.paper], [3.0, 1.05, M.paper]])
  box(x, y, L - 0.02, 0.06, 0.06, 0.004, c);
box(2.76, 1.44, L - 0.1, 0.26, 0.03, 0.05, M.steel); box(2.87, 1.47, L - 0.06, 0.04, 0.07, 0.06, M.steel);   // picture light
// Storage bench under the window, between the bed and the desk.
box(1.72, 0, L - 0.6, 0.62, 0.36, 0.45, M.plum); box(1.74, 0.06, L - 0.61, 0.58, 0.2, 0.012, M.white);
box(1.7, 0.36, L - 0.62, 0.66, 0.1, 0.48, M.olive); box(1.82, 0.46, L - 0.52, 0.3, 0.012, 0.22, M.red);

// ---------- desk in the corner of the desk wall and the curtain wall ----------
const dz = L - 1.05, dd = 0.55;
box(W - dd, 0.7, dz, dd, 0.09, 1.05, M.petals);                                    // thick laminate top
box(W - dd + 0.03, 0, dz + 0.02, dd - 0.06, 0.7, 0.03, M.plum);                    // side panel
box(W - dd + 0.06, 0.38, dz + 0.05, dd - 0.1, 0.03, 0.98, M.plum);                 // shelf under the top
box(W - 0.3, 0.79, dz + 0.42, 0.18, 0.015, 0.22, M.black);                         // monitor foot
box(W - 0.22, 0.8, dz + 0.51, 0.03, 0.12, 0.04, M.black);
box(W - 0.24, 0.9, dz + 0.2, 0.025, 0.36, 0.62, M.black);                          // monitor
{ const towel = new THREE.Mesh(new THREE.SphereGeometry(0.5, 14, 10), M.pink);
  towel.scale.set(0.34, 0.16, 0.42); towel.position.set(W - 0.36, 0.85, dz + 0.72); room.add(towel);
  const bottle = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.26, 14), M.white);
  bottle.position.set(W - 0.42, 0.92, dz + 0.12); room.add(bottle); }
box(W - 0.48, 0, dz + 0.3, 0.36, 0.2, 0.5, M.red);                                 // crate under the desk

// ---------- desk wall: switches, tube light; floor: laundry basket, backpack ----------
box(W - 0.012, 1.18, 1.55, 0.012, 0.09, 0.22, M.white);
{ const cable = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
    new THREE.Vector3(W - 0.02, 1.18, 1.6), new THREE.Vector3(W - 0.03, 0.6, 1.5), new THREE.Vector3(W - 0.08, 0.05, 1.2),
    new THREE.Vector3(W - 0.5, 0.02, 0.9)]), 24, 0.006, 6), M.white); room.add(cable); }
{ const tube = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 1.2, 10), M.glow);
  tube.rotation.x = Math.PI / 2; tube.position.set(W - 0.04, 2.42, 2.1); room.add(tube);
  const lamp = new THREE.PointLight(0xfff6e8, 9, 0, 1.6); lamp.position.set(W - 0.35, 2.3, 2.0);
  lamp.castShadow = true; lamp.shadow.mapSize.set(1024, 1024); lamp.shadow.bias = -0.002; room.add(lamp); }
{ const basket = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.17, 0.42, 20, 1, true), mat(0x6f7072, { side: THREE.DoubleSide }));
  basket.position.set(2.42, 0.21, 2.35); basket.castShadow = true; room.add(basket);
  const pack = new THREE.Mesh(new THREE.SphereGeometry(0.5, 14, 10), M.navy);
  pack.scale.set(0.3, 0.46, 0.36); pack.position.set(2.72, 0.23, 1.95); pack.castShadow = true; room.add(pack); }

scene.add(new THREE.HemisphereLight(0xffffff, 0xb9b2a4, 1.35));
const fill = new THREE.PointLight(0xffffff, 4, 0, 1.6); fill.position.set(1.2, 2.5, 1.8); room.add(fill);

// ---------- views: where the phone stood in four frames of the video, and two overviews ----------
const VIEWS = {
  "Dollhouse": { from: [5.9, 5.0, 6.4], to: [W / 2, 0.9, L / 2], fov: 40, open: true },
  "From above": { from: [W / 2, 8.2, L / 2 + 0.01], to: [W / 2, 0, L / 2], fov: 36, open: true },
  "From the desk (frame 25)": { from: [2.75, 1.5, 3.2], to: [1.0, 1.15, 0], fov: 69 },
  "From the door (frame 13)": { from: [2.55, 1.5, 0.75], to: [2.35, 0.95, L], fov: 69 },
  "Across the bed (frame 10)": { from: [2.3, 1.55, 1.3], to: [0.35, 1.1, L], fov: 69 },
  "Bedside (frame 12)": { from: [1.5, 1.35, 2.7], to: [0, 1.25, 1.15], fov: 69 },
};
function setView(name) {
  const v = VIEWS[name] || VIEWS["Dollhouse"];
  camera.fov = v.fov; camera.position.set(W - v.from[0], v.from[1], v.from[2]);
  controls.target.set(W - v.to[0], v.to[1], v.to[2]);
  ceiling.visible = !v.open; camera.updateProjectionMatrix(); controls.update();
}
const bar = document.getElementById("bar");
for (const name of Object.keys(VIEWS)) { const b = document.createElement("button"); b.textContent = name;
  b.onclick = () => setView(name); bar.appendChild(b); }
setView(params.get("view") || "Dollhouse");

addEventListener("resize", () => { renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); });
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
window.roomReady = true;
