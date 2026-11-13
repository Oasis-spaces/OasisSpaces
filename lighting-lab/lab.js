// The lighting lab: the hand-built bedroom (handbuilt-room/, every surface photographed from
// the video) lit by physics instead of by the light that happened to be there when it was filmed.
//
// - The filmed shading is taken out of each photograph (its broad light-and-shade divided away,
//   the pattern kept), so what is left is the surface's own colour: its albedo.
// - Lamps are real bulbs with lumens, watts, colour temperature and power factor, on a mains
//   supply whose voltage and dimmer change what they give (physics.js, bulbState).
// - The sun is placed for a date, time and place, and comes in only through the window opening;
//   the sky comes in as the window's glow, or through the drawn curtain, tinted by it.
// - Bounced light is solved by radiosity on ~20 cm patches and drawn as light maps; direct light
//   from lamps and sun is drawn by three.js with shadow maps, so shadows stay sharp where they should.
// - The camera adapts like an eye to the room's average luminance; bright things bloom (glare).
// - Click any surface for its lux; the panel gives power, current, energy, cost and glare.
//
// Plan coordinates as in handbuilt-room/room.json: x from the headboard wall to the desk wall,
// z from the wardrobe wall to the curtain wall, y up, metres.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { sunPosition, daylight, kelvinToRGB, BULBS, bulbState, Radiosity, ugr, ugrWords, DEG } from "./physics.js";

const params = new URLSearchParams(location.search);
const ROOM = "../handbuilt-room/";
const MIN_SEEN = 0.45;
const PLAIN = { floor: "#cfc8b6", ceiling: "#efefec" };
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- renderer
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.AgXToneMapping;
$("view").appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0c0e);
const camera = new THREE.PerspectiveCamera(69, innerWidth / innerHeight, 0.03, 60);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
// The eye's exposure is applied before the bloom, so the bloom sees display brightness, and
// anything over 32x white (the sun's patch, a bare filament) is held there: past that it is all white.
const exposurePass = new ShaderPass({
  uniforms: { tDiffuse: { value: null }, exposure: { value: 1 } },
  vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
  fragmentShader: "uniform sampler2D tDiffuse; uniform float exposure; varying vec2 vUv;\n" +
    "void main() { vec4 c = texture2D(tDiffuse, vUv); gl_FragColor = vec4(min(c.rgb * exposure, vec3(32.0)), c.a); }",
});
composer.addPass(exposurePass);
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.2, 0.15, 1);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// ---------------------------------------------------------------- the room
const [room, coverage] = await Promise.all([
  fetch(ROOM + "room.json").then((r) => r.json()),
  fetch(ROOM + "textures/coverage.json").then((r) => r.json()).catch(() => ({})),
]);
const { W, L, H } = room.size;
const group = new THREE.Group(); group.scale.x = -1; group.position.x = W; scene.add(group);

// The opening behind the curtains. Not measured: the curtains hang from x 0.06 to 1.81 m and
// from a rod at 1.94 m, and behind them is a glazed door or window, assumed to fill that span.
const WINDOW = { x0: 0.15, x1: 1.75, y0: 0.3, y1: 2.0, z: L };

const v3 = (a) => new THREE.Vector3(...a);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

// What each thing is made of: how glossy it is. Reflectance comes from its photograph.
const KINDS = {
  paint:    { label: "Matt paint",      roughness: 0.92 },
  tile:     { label: "Glazed tile",     roughness: 0.22 },
  laminate: { label: "Laminate",        roughness: 0.45 },
  wood:     { label: "Wood",            roughness: 0.6 },
  fabric:   { label: "Fabric",          roughness: 0.97, sheen: 0.6 },
  screen:   { label: "Glass screen",    roughness: 0.06 },
};
function kindOf(name) {
  if (name.startsWith("floor")) return "tile";
  if (name.startsWith("ceiling") || name.startsWith("wall")) return "paint";
  if (name.startsWith("bed")) return "fabric";
  if (name.startsWith("monitor")) return "screen";
  if (name.startsWith("bench")) return "wood";
  return "laminate";
}

// Faces: every surface light lands on. Shell surfaces face into the room; pieces face out.
const faces = [];
const centre = [W / 2, H / 2, L / 2];
function addFace(def) {
  let n = cross(def.U, def.V);
  const mid = [0, 1, 2].map((k) => def.O[k] + def.U[k] * def.w / 2 + def.V[k] * def.h / 2);
  const toCentre = [0, 1, 2].map((k) => centre[k] - mid[k]);
  const out = def.inward ? dot(n, toCentre) < 0 : def.boxCentre && dot(n, [0, 1, 2].map((k) => mid[k] - def.boxCentre[k])) < 0;
  if (out) n = n.map((v) => -v);
  def.n = n;
  def.kind = def.kind || kindOf(def.name);
  faces.push(def);
  return def;
}
function sub(def, a0, a1, b0, b1, name, extra = {}) {
  const O = [0, 1, 2].map((k) => def.O[k] + def.U[k] * a0 + def.V[k] * b0);
  return { ...def, ...extra, name, O, w: a1 - a0, h: b1 - b0,
    uv: [a0 / def.w, a1 / def.w, b0 / def.h, b1 / def.h] };
}
for (const s of room.surfaces.filter((s) => !s.of)) {
  const def = { name: s.name, O: s.origin, U: s.u, V: s.v, w: s.w, h: s.h, inward: true, tex: s.name, uv: [0, 1, 0, 1] };
  if (s.name === "wall-curtain") {
    // Cut the opening out; a = distance along the wall from its origin (x = W, going to x = 0).
    const a0 = W - WINDOW.x1, a1 = W - WINDOW.x0;
    addFace(sub(def, 0, s.w, 0, WINDOW.y0, "wall-curtain"));
    addFace(sub(def, 0, s.w, WINDOW.y1, s.h, "wall-curtain (above)"));
    addFace(sub(def, 0, a0, WINDOW.y0, WINDOW.y1, "wall-curtain (desk side)"));
    addFace(sub(def, a1, s.w, WINDOW.y0, WINDOW.y1, "wall-curtain (bed side)"));
    WINDOW.face = sub(def, a0, a1, WINDOW.y0, WINDOW.y1, "window");
  } else addFace(def);
}
const boxes = room.boxes.filter((b) => b.draw !== false);
const near = (a, b) => Math.abs(a - b) < 0.02;
for (const b of boxes) {
  const c = b.min.map((v, k) => (v + b.max[k]) / 2), size = b.max.map((v, k) => v - b.min[k]);
  for (let k = 0; k < 3; k++) for (const side of [0, 1]) {
    const at = side ? b.max[k] : b.min[k];
    if (k === 1 && side === 0 && at < 0.01) continue;                       // standing on the floor
    if (k !== 1 && (near(at, 0) || near(at, k === 0 ? W : L))) continue;   // against a wall
    if (k === 1 && side === 1 && near(at, H)) continue;
    // Is there a photographed surface for this side of this piece?
    const s = room.surfaces.find((s) => s.of === b.name && near(s.origin[k], at)
      && Math.abs(s.u[k]) < 0.5 && Math.abs(s.v[k]) < 0.5);
    const i = (k + 1) % 3, j = (k + 2) % 3;
    const fits = s && near(s.w * s.h, size[i] * size[j]);
    const O = [0, 0, 0]; O[k] = at; O[i] = b.min[i]; O[j] = b.min[j];
    const U = [0, 0, 0]; U[i] = 1; const V = [0, 0, 0]; V[j] = 1;
    const def = fits ? { name: s.name, O: s.origin, U: s.u, V: s.v, w: s.w, h: s.h, tex: s.name, uv: [0, 1, 0, 1] }
      : { name: `${b.name}-${"xyz"[k]}${side ? "+" : "-"}`, O, U, V, w: size[i], h: size[j], color: b.color, uv: [0, 1, 0, 1] };
    def.boxCentre = c; def.owner = b.name;
    if (k === 1 && side === 1) def.top = true;
    addFace(def);
  }
}

// ---------------------------------------------------------------- photographs -> albedo
const toLin = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const toSrgb = (c) => (c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
function boxBlur(src, w, h, r) {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) { let s = 0, n = 0;
    for (let x = -r; x < w + r; x++) {
      if (x + r < w && x + r >= 0) { s += src[y * w + x + r]; n++; }
      if (x - r - 1 >= 0 && x - r - 1 < w) { s -= src[y * w + x - r - 1]; n--; }
      if (x >= 0 && x < w) tmp[y * w + x] = s / n; } }
  for (let x = 0; x < w; x++) { let s = 0, n = 0;
    for (let y = -r; y < h + r; y++) {
      if (y + r < h && y + r >= 0) { s += tmp[(y + r) * w + x]; n++; }
      if (y - r - 1 >= 0 && y - r - 1 < h) { s -= tmp[(y - r - 1) * w + x]; n--; }
      if (y >= 0 && y < h) out[y * w + x] = s / n; } }
  return out;
}
// Divide out the broad light and shade the camera saw (a blur a sixth of the surface wide),
// keep the pattern, keep the average: a first, classical estimate of the surface's own colour.
// A learned intrinsic-image model replaces this in the pipeline (see docs/realism-lighting-plan).
async function photo(name) {
  const img = new Image(); img.src = `${ROOM}textures/${name}.jpg`; await img.decode();
  const scale = Math.min(1, 512 / Math.max(img.width, img.height));
  const w = Math.round(img.width * scale), h = Math.round(img.height * scale);
  const c1 = document.createElement("canvas"); c1.width = w; c1.height = h;
  const g1 = c1.getContext("2d", { willReadFrequently: true }); g1.drawImage(img, 0, 0, w, h);
  const data = g1.getImageData(0, 0, w, h);
  const lin = new Float32Array(w * h * 3), luma = new Float32Array(w * h);
  for (let p = 0; p < w * h; p++) {
    for (let c = 0; c < 3; c++) lin[p * 3 + c] = toLin(data.data[p * 4 + c] / 255);
    luma[p] = 0.2126 * lin[p * 3] + 0.7152 * lin[p * 3 + 1] + 0.0722 * lin[p * 3 + 2];
  }
  const r = Math.max(4, Math.round(Math.max(w, h) / 6));
  const low = boxBlur(boxBlur(luma, w, h, r), w, h, r);
  let mean = 0; for (let p = 0; p < w * h; p++) mean += luma[p]; mean /= w * h;
  const out = new ImageData(w, h), delit = new Float32Array(w * h * 3);
  for (let p = 0; p < w * h; p++) {
    const k = mean / Math.max(low[p], 0.02);
    for (let c = 0; c < 3; c++) {
      delit[p * 3 + c] = Math.min(0.95, lin[p * 3 + c] * k);
      out.data[p * 4 + c] = Math.round(toSrgb(delit[p * 3 + c]) * 255);
    }
    out.data[p * 4 + 3] = 255;
  }
  const c2 = document.createElement("canvas"); c2.width = w; c2.height = h; c2.getContext("2d").putImageData(out, 0, 0);
  const tex = (canvas) => { const t = new THREE.CanvasTexture(canvas); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t; };
  // Mean colour of a part of the image (uv rectangle, v up).
  const meanOf = (arr, [u0, u1, v0, v1]) => {
    const x0 = Math.floor(u0 * w), x1 = Math.max(x0 + 1, Math.floor(u1 * w));
    const y0 = Math.floor((1 - v1) * h), y1 = Math.max(y0 + 1, Math.floor((1 - v0) * h));
    const m = [0, 0, 0]; let n = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { for (let c = 0; c < 3; c++) m[c] += arr[(y * w + x) * 3 + c]; n++; }
    return m.map((v) => v / n);
  };
  return { filmed: tex(c1), delit: tex(c2), meanOf: (uv, delitOn) => meanOf(delitOn ? delit : lin, uv) };
}
const photos = {};
await Promise.all([...new Set(faces.concat([WINDOW.face]).map((f) => f.tex).filter(Boolean))].map(async (name) => {
  const seen = coverage[name] ?? 0;
  if (name !== "ceiling" && (seen >= MIN_SEEN || name === "wall-wardrobe" || name === "wall-curtain")) photos[name] = await photo(name);
}));

// ---------------------------------------------------------------- meshes
const state = {
  delit: true, heat: false, glare: true, ev: 0,
  // Placeholder place and time (editable in the panel): the flat's location is not known.
  lat: 28.61, lon: 77.21, utc: 5.5, date: new Date().toISOString().slice(0, 10), minutes: 10 * 60,
  facing: 180, sky: "clear", curtain: "open",
  volts: 230, tariff: 7, currency: "₹", co2: 0.71,
  lamps: [],
};
let nextLamp = 1;
function quad(f) {
  // Triangles wound so that U x V points along the face's normal.
  let [U, V, w, h, uv] = [f.U, f.V, f.w, f.h, f.uv];
  const flip = dot(cross(U, V), f.n) < 0;
  const corner = (a, b) => [0, 1, 2].map((k) => f.O[k] + U[k] * a + V[k] * b);
  const pts = [corner(0, 0), corner(w, 0), corner(w, h), corner(0, h)];
  const uvs = [[uv[0], uv[2]], [uv[1], uv[2]], [uv[1], uv[3]], [uv[0], uv[3]]];
  const lm = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pts.flat(), 3));
  g.setAttribute("normal", new THREE.Float32BufferAttribute([...f.n, ...f.n, ...f.n, ...f.n], 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uvs.flat(), 2));
  g.setAttribute("uv1", new THREE.Float32BufferAttribute(lm.flat(), 2));
  g.setIndex(flip ? [0, 2, 1, 0, 3, 2] : [0, 1, 2, 0, 2, 3]);
  return g;
}
const meshes = [];
let ceilingMeshes = [];
const rad = new Radiosity(faces, boxes.map((b) => ({ min: b.min, max: b.max })), 0.2);
for (const f of faces) {
  const k = KINDS[f.kind];
  const mat = new THREE.MeshPhysicalMaterial({ roughness: k.roughness, metalness: 0, sheen: k.sheen || 0,
    sheenRoughness: 0.8, side: THREE.FrontSide, shadowSide: THREE.DoubleSide });
  f.photo = f.tex && photos[f.tex];
  if (!f.photo) mat.color.set(f.color || PLAIN[f.name] || "#e9e8e3");
  f.gain = 1;
  const lm = new THREE.DataTexture(new Uint16Array(f.nu * f.nv * 4), f.nu, f.nv, THREE.RGBAFormat, THREE.HalfFloatType);
  lm.magFilter = lm.minFilter = THREE.LinearFilter; lm.colorSpace = THREE.LinearSRGBColorSpace;
  const heat = new THREE.DataTexture(new Uint8Array(f.nu * f.nv * 4), f.nu, f.nv);
  heat.magFilter = THREE.LinearFilter; heat.colorSpace = THREE.SRGBColorSpace;
  mat.lightMap = lm;
  f.lit = mat; f.heatTex = heat;
  f.heatMat = new THREE.MeshBasicMaterial({ map: heat, toneMapped: false });
  const mesh = new THREE.Mesh(quad(f), mat);
  mesh.castShadow = mesh.receiveShadow = true;
  mesh.userData.face = f; f.mesh = mesh;
  group.add(mesh); meshes.push(mesh);
  if (f.name === "ceiling") ceilingMeshes.push(mesh);
}
// The shell's walls are paper-thin, and light leaks through their seams in a shadow map. Thick,
// invisible slabs just outside them cast the shadows instead (the window opening left open).
{
  const T = 0.3, slab = (min, max) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(...max.map((v, k) => v - min[k])),
      new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false, shadowSide: THREE.FrontSide }));
    m.position.set(...min.map((v, k) => (v + max[k]) / 2)); m.castShadow = true; group.add(m);
  };
  const e = 0.005;    // 5 mm into the room, so slabs overlap at the corners; less than the shadow normal bias
  slab([-T, -T, -T], [W + T, e, L + T]); slab([-T, H - e, -T], [W + T, H + T, L + T]);
  slab([-T, 0, -T], [e, H, L + T]); slab([W - e, 0, -T], [W + T, H, L + T]); slab([-T, 0, -T], [W + T, H, e]);
  slab([-T, 0, L - e], [W + T, WINDOW.y0, L + T]); slab([-T, WINDOW.y1, L - e], [W + T, H, L + T]);
  slab([-T, WINDOW.y0, L - e], [WINDOW.x0, WINDOW.y1, L + T]); slab([WINDOW.x1, WINDOW.y0, L - e], [W + T, WINDOW.y1, L + T]);
  for (const m of meshes) if (!m.userData.face.owner) m.castShadow = false;
}
function setAlbedo() {
  for (const f of faces) {
    if (f.photo) {
      f.lit.map = state.delit ? f.photo.delit : f.photo.filmed; f.lit.needsUpdate = true;
      f.rho = f.photo.meanOf(f.uv, state.delit).map((v) => Math.min(0.95, v * f.gain));
      f.lit.color.setScalar(f.gain);
    } else {
      const c = new THREE.Color(f.color || PLAIN[f.name] || "#e9e8e3");
      f.rho = [c.r, c.g, c.b].map((v) => Math.min(0.95, v * f.gain));
      f.lit.color.setRGB(...f.rho, THREE.LinearSRGBColorSpace);
    }
  }
  rad.updateReflectance();
}
setAlbedo();

// The opening: the curtain's photograph glowing with what comes through it, or, drawn back,
// a plain panel as bright as the sky behind the glass (no invented view).
const win = WINDOW.face;
win.n = [0, 0, -1];
const curtainMean = photos["wall-curtain"]?.meanOf(win.uv, false) || [0.3, 0.4, 0.4];
const winMat = new THREE.MeshPhysicalMaterial({ roughness: 0.97, sheen: 0.6, map: photos["wall-curtain"]?.filmed,
  emissive: 0xffffff, emissiveMap: photos["wall-curtain"]?.filmed, side: THREE.FrontSide });
const skyMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
const winMesh = new THREE.Mesh(quad(win), winMat);
winMesh.receiveShadow = true;
group.add(winMesh);

// ---------------------------------------------------------------- lamps
const FIXTURES = {
  ceiling: { label: "Ceiling panel", fixture: "panel", bulb: "panel-15" },
  bulb:    { label: "Bare bulb (holder)", fixture: "bare", bulb: "led-9" },
  tube:    { label: "Tube light on a wall", fixture: "panel", bulb: "tube-20" },
  table:   { label: "Table lamp, shaded", fixture: "shade", bulb: "led-5" },
};
function addLamp(type, pos, axis = [0, -1, 0], extra = {}) {
  const F = FIXTURES[type];
  const lamp = { id: nextLamp++, type, name: extra.name || F.label, fixture: F.fixture, bulb: extra.bulb || F.bulb,
    pos, axis, cone: 40, on: extra.on ?? true, dim: 1, hours: extra.hours ?? 5 };
  lamp.obj = makeLampObject(lamp);
  state.lamps.push(lamp);
  return lamp;
}
function makeLampObject(lamp) {
  const g = new THREE.Group();
  g.position.fromArray(lamp.pos);
  let light, extra = null, emitter, area;
  const ax = v3(lamp.axis);
  if (lamp.fixture === "bare") {
    light = new THREE.PointLight(0xffffff, 1, 0, 2);
    emitter = new THREE.Mesh(new THREE.SphereGeometry(0.03, 20, 12), new THREE.MeshBasicMaterial());
    area = Math.PI * 0.03 ** 2;
  } else if (lamp.fixture === "panel") {
    light = new THREE.SpotLight(0xffffff, 1, 0, Math.PI / 2 * 0.96, 1, 2);
    const tube = lamp.bulb.startsWith("tube");
    const geo = tube ? new THREE.CylinderGeometry(0.015, 0.015, 1.2, 12) : new THREE.CircleGeometry(0.15, 32);
    emitter = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
    if (tube) emitter.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), Math.abs(ax.y) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0).cross(ax).normalize());
    else emitter.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), ax);
    area = tube ? 1.2 * 0.03 : Math.PI * 0.15 ** 2;
  } else {
    light = new THREE.SpotLight(0xffffff, 1, 0, lamp.cone * DEG, 0.35, 2);
    extra = new THREE.PointLight(0xffffff, 1, 0, 2);
    const shade = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.14, 0.18, 32, 1, true),
      new THREE.MeshStandardMaterial({ color: 0xe8e0d0, roughness: 0.9, side: THREE.DoubleSide, emissive: 0xffffff }));
    shade.position.y = -0.02;
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.4), new THREE.MeshStandardMaterial({ color: 0x333333, roughness: 0.4, metalness: 0.8 }));
    stem.position.y = -0.27;
    g.add(shade, stem);
    g.userData.shade = shade;
    emitter = new THREE.Mesh(new THREE.SphereGeometry(0.025, 16, 10), new THREE.MeshBasicMaterial());
    area = 0;
  }
  light.castShadow = true;
  light.shadow.mapSize.set(1024, 1024);
  light.shadow.bias = -0.0008; light.shadow.normalBias = 0.02;
  light.shadow.camera.near = 0.05;
  g.add(light, emitter);
  if (extra) g.add(extra);
  if (light.target) { light.target.position.copy(ax); g.add(light.target); }
  group.add(g);
  return { group: g, light, extra, emitter, area };
}
function removeLamp(lamp) {
  group.remove(lamp.obj.group);
  lamp.obj.group.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); if (o.shadow?.map) o.shadow.map.dispose(); });
  state.lamps = state.lamps.filter((l) => l !== lamp);
}
// Starting lamps: placeholders, since the video's own fittings are not detected yet.
addLamp("ceiling", [W / 2, H - 0.02, L / 2], [0, -1, 0], { name: "Ceiling light", hours: 5 });
addLamp("table", [2.3, 1.25, 3.13], [0, -1, 0], { name: "Desk lamp", hours: 3 });

// ---------------------------------------------------------------- sun and sky
const sun = new THREE.DirectionalLight(0xffffff, 0);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -5, right: 5, top: 5, bottom: -5, near: 0.1, far: 30 });
sun.shadow.bias = -0.0003; sun.shadow.normalBias = 0.012;
group.add(sun, sun.target);
sun.target.position.set(W / 2, 0, L / 2);
const GLASS = 0.8, CURTAIN = 0.12, GROUND = 0.2;

function instant() {
  const [y, m, d] = state.date.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d) + (state.minutes - state.utc * 60) * 60000);
}
function outside() {
  const { altitude, azimuth } = sunPosition(instant(), state.lat, state.lon);
  const day = daylight(altitude, state.sky);
  // Towards the sun, in plan coordinates: +z is the way the window faces, +x is 90° clockwise of it.
  const rel = (azimuth - state.facing) * DEG, alt = altitude * DEG;
  const dir = [Math.cos(alt) * Math.sin(rel), Math.sin(alt), Math.cos(alt) * Math.cos(rel)];
  const skyVertical = 0.5 * day.skyHorizontal + 0.5 * GROUND * (day.skyHorizontal + day.sunNormal * Math.max(0, Math.sin(alt)));
  const sunOnWindow = day.sunNormal * Math.max(0, dir[2]);
  const sunK = 2400 + 3400 * Math.min(1, Math.max(0, Math.sin(alt)) / 0.5);
  return { altitude, azimuth, dir, day, skyVertical, sunOnWindow, sunRGB: kelvinToRGB(sunK), sunK };
}

// ---------------------------------------------------------------- solve and draw
let out = null, adaptL = 1, exposure = 1;
let metered = 0, viewL = null, panelPart = null;
const panelDirect = (p, c) => (panelPart ? panelPart[p * 3 + c] : 0);
const heatStops = ["#0d0887", "#5c01a6", "#9c179e", "#cc4778", "#ed7953", "#fdb42f", "#f0f921"].map((c) => new THREE.Color(c));
function heatColour(lux) {
  const t = Math.min(1, Math.max(0, Math.log10(Math.max(lux, 1)) / Math.log10(3000)));
  const x = t * (heatStops.length - 1), i = Math.min(heatStops.length - 2, Math.floor(x));
  return heatStops[i].clone().lerp(heatStops[i + 1], x - i);
}
function relight() {
  out = outside();
  const night = out.altitude <= 0;
  // Lamps: what each bulb gives at this voltage and dimmer setting.
  const lampSources = [];
  for (const lamp of state.lamps) {
    const s = bulbState(BULBS[lamp.bulb], state.volts, lamp.dim);
    lamp.now = s;
    const lm = lamp.on ? s.lumens : 0, rgb = kelvinToRGB(s.cct);
    const { light, extra, emitter, area } = lamp.obj;
    light.color.setRGB(...rgb, THREE.LinearSRGBColorSpace);
    light.visible = lm > 0 && lamp.fixture !== "panel";   // a panel's soft light is drawn from the radiosity
    if (lamp.fixture === "bare") light.intensity = lm / (4 * Math.PI);
    else if (lamp.fixture === "panel") light.intensity = lm / Math.PI;
    else {
      const cc = Math.cos(lamp.cone * DEG);
      light.intensity = (0.85 * lm) / (2 * Math.PI * (1 - cc));
      extra.color.copy(light.color); extra.intensity = (0.15 * lm) / (4 * Math.PI); extra.visible = lm > 0;
      lamp.obj.group.userData.shade.material.emissiveIntensity = (0.15 * lm) / (4 * Math.PI) / 0.06;   // its glow, in nits
      lamp.obj.group.userData.shade.material.emissive.setRGB(...rgb, THREE.LinearSRGBColorSpace);
    }
    // The bulb's own brightness (nits): what you see when you look at it.
    const Lb = lamp.fixture === "bare" ? lm / (4 * Math.PI) / area : lamp.fixture === "panel" ? lm / (Math.PI * area) : lm / (4 * Math.PI) / (Math.PI * 0.025 ** 2);
    lamp.luminance = Lb;
    emitter.material.color.setRGB(...rgb.map((c) => Math.max(c * Lb, 0.02)), THREE.LinearSRGBColorSpace);
    if (lm > 0) lampSources.push({ pos: lamp.pos, fixture: lamp.fixture, lumens: lm, rgb, axis: lamp.axis, cone: lamp.cone });
  }
  // Daylight: the sun only through the open window; the sky as the opening's glow.
  const open = state.curtain === "open";
  const skyRGB = night ? kelvinToRGB(9000) : kelvinToRGB(state.sky === "overcast" ? 6500 : 7500);
  let winL, winRGB;
  if (open) { winL = (GLASS * out.skyVertical) / Math.PI; winRGB = skyRGB; }
  else {
    winL = (GLASS * CURTAIN * (out.skyVertical + out.sunOnWindow)) / Math.PI;
    const y = 0.2126 * curtainMean[0] + 0.7152 * curtainMean[1] + 0.0722 * curtainMean[2];
    winRGB = curtainMean.map((c, k) => (c / y) * (out.sunOnWindow > out.skyVertical ? out.sunRGB[k] : skyRGB[k]));
  }
  const sunLux = open && out.day.sunNormal > 0 && out.dir[2] > 0.001 ? GLASS * out.day.sunNormal : 0;
  sun.intensity = sunLux;
  sun.visible = sunLux > 0;
  sun.color.setRGB(...out.sunRGB, THREE.LinearSRGBColorSpace);
  sun.position.set(W / 2 + out.dir[0] * 10, out.dir[1] * 10, L / 2 + out.dir[2] * 10);
  out.sunLux = sunLux; out.winL = winL;

  // Light maps carry the panels' direct light too: solve them alone once more for that part.
  const panels = lampSources.filter((l) => l.fixture === "panel");
  panelPart = null;
  if (panels.length) {
    panelPart = new Float32Array(rad.n * 3);
    for (let i = 0; i < rad.n; i++) {
      const d = rad.directAt(rad.pos[i * 3], rad.pos[i * 3 + 1], rad.pos[i * 3 + 2], rad.nrm[i * 3], rad.nrm[i * 3 + 1], rad.nrm[i * 3 + 2], { lamps: panels });
      for (let c = 0; c < 3; c++) panelPart[i * 3 + c] = d[0][c];
    }
  }
  rad.solve({ lamps: lampSources, sun: sunLux ? { dir: out.dir, lux: sunLux, rgb: out.sunRGB } : null,
    window: { ...WINDOW, luminance: winL, rgb: winRGB } });

  // Light maps: everything three.js does not draw itself (the sky's glow and every bounce).
  for (const f of faces) {
    const lm = f.lit.lightMap.image.data, hm = f.heatTex.image.data;
    for (let j = 0; j < f.nv; j++) for (let i = 0; i < f.nu; i++) {
      const p = f.first + j * f.nu + i, t = (j * f.nu + i) * 4;
      let total = 0;
      for (let c = 0; c < 3; c++) {
        const v = panelDirect(p, c) + rad.parts[p * 9 + 6 + c] + rad.indirect[p * 3 + c];
        lm[t + c] = THREE.DataUtils.toHalfFloat(v);
        total += [0.2126, 0.7152, 0.0722][c] * (rad.direct[p * 3 + c] + rad.indirect[p * 3 + c]);
      }
      lm[t + 3] = THREE.DataUtils.toHalfFloat(1);
      const col = heatColour(total);
      hm[t] = col.r * 255; hm[t + 1] = col.g * 255; hm[t + 2] = col.b * 255; hm[t + 3] = 255;
    }
    f.lit.lightMap.needsUpdate = true; f.heatTex.needsUpdate = true;
  }
  // The opening.
  if (open) { winMesh.material = skyMat; skyMat.color.setRGB(...winRGB.map((c) => c * winL), THREE.LinearSRGBColorSpace); }
  else {
    winMesh.material = winMat;
    const y = 0.2126 * curtainMean[0] + 0.7152 * curtainMean[1] + 0.0722 * curtainMean[2];
    winMat.emissiveIntensity = winL / Math.max(y, 0.05);
    winMat.emissive.setRGB(...(out.sunOnWindow > out.skyVertical ? out.sunRGB : skyRGB), THREE.LinearSRGBColorSpace);
    winMat.map = state.delit ? photos["wall-curtain"]?.delit : photos["wall-curtain"]?.filmed;
    winMat.lightMap = null; winMat.needsUpdate = true;
  }
  adaptL = Math.max(rad.adaptation(), 0.5);      // the room's own log-average: the background for glare
  metered = 0;                                      // the eye re-meters the view (see meter())
  renderer.toneMappingExposure = 1;
  bloom.highPassUniforms.luminosityThreshold.value = 1.5;
  for (const m of meshes) m.material = state.heat ? m.userData.face.heatMat : m.userData.face.lit;
  for (const lamp of state.lamps) lamp.obj.group.visible = !state.heat;
  winMesh.visible = !state.heat;
  report();
}

// ---------------------------------------------------------------- readouts
const fmt = (v, d = 0) => v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
function report() {
  const o = out;
  $("sunline").textContent = o.altitude > 0
    ? `Sun ${fmt(o.altitude)}° up, bearing ${fmt(o.azimuth)}°. Outside: ${fmt(o.day.sunNormal / 1000, 0)} klx direct, ${fmt(o.day.skyHorizontal / 1000, 1)} klx from the sky.`
    : `Sun ${fmt(-o.altitude)}° below the horizon. Outside: ${fmt(o.day.skyHorizontal, 1)} lx.`;
  $("sunin").textContent = o.sunLux > 0 ? `Sunlight comes through the window (${fmt(o.sunLux / 1000, 0)} klx in the beam).`
    : state.curtain === "closed" && o.sunOnWindow > 0 ? "Sun on the curtains: they glow, tinted by their colour." : o.altitude > 0 ? "No direct sun on this window now." : "";
  // Energy.
  let watts = 0, amps = 0, va = 0, dayWh = 0, heat = 0, ledSave = 0;
  const list = $("lamps"); list.innerHTML = "";
  for (const lamp of state.lamps) {
    const s = lamp.now, b = BULBS[lamp.bulb], on = lamp.on && s.watts > 0;
    if (on) { watts += s.watts; amps += s.amps; va += s.va; heat += s.watts - s.lumens / 300; }
    dayWh += (lamp.on ? s.watts : 0) * lamp.hours;
    if (b.tech !== "led" && lamp.on) ledSave += (s.watts - s.lumens / 100) * lamp.hours;
    list.appendChild(lampRow(lamp));
  }
  const kwhMonth = (dayWh * 30) / 1000;
  $("energy").innerHTML = `
    <div class="stat"><b>${fmt(watts, 1)} W</b><span>drawn now</span></div>
    <div class="stat"><b>${fmt(amps, 2)} A</b><span>at ${state.volts} V (${fmt(va, 0)} VA)</span></div>
    <div class="stat"><b>${fmt(kwhMonth, 1)} kWh</b><span>a month at these hours</span></div>
    <div class="stat"><b>${state.currency}${fmt(kwhMonth * state.tariff, 0)}</b><span>a month · ${fmt(kwhMonth * state.co2, 1)} kg CO₂</span></div>
    <p class="fine">≈${fmt(heat, 0)} W of it ends up as heat in the room.${ledSave > 1 ? ` Swapping the filament and CFL bulbs for LEDs of the same light saves ${fmt((ledSave * 30) / 1000, 1)} kWh (${state.currency}${fmt((ledSave * 30 / 1000) * state.tariff, 0)}) a month.` : ""}</p>`;
  glareReport();
}
function lampRow(lamp) {
  const row = document.createElement("div"); row.className = "lamp";
  const s = lamp.now;
  const opts = Object.entries(BULBS).map(([k, b]) => `<option value="${k}" ${k === lamp.bulb ? "selected" : ""}>${b.name}</option>`).join("");
  row.innerHTML = `
    <div class="lamphead"><label><input type="checkbox" ${lamp.on ? "checked" : ""}> ${lamp.name}</label>
      <button class="del" title="Remove">✕</button></div>
    <select>${opts}</select>
    <label class="slider">Dimmer <input type="range" min="0.05" max="1" step="0.01" value="${lamp.dim}"><output>${Math.round(lamp.dim * 100)}%</output></label>
    <label class="slider">On for <input type="number" min="0" max="24" step="0.5" value="${lamp.hours}"> h a day</label>
    <div class="fine">${lamp.on ? `${fmt(s.lumens)} lm · ${fmt(s.watts, 1)} W · ${fmt(s.amps, 3)} A · ${fmt(s.cct)} K · ${fmt(s.lumens / Math.max(s.watts, 0.01))} lm/W` : "off"}${s.note ? ` · <em>${s.note}</em>` : ""}</div>`;
  const [check] = row.querySelectorAll("input[type=checkbox]");
  check.onchange = () => { lamp.on = check.checked; schedule(); };
  row.querySelector("select").onchange = (e) => { lamp.bulb = e.target.value; schedule(); };
  const dim = row.querySelector("input[type=range]");
  dim.oninput = () => { lamp.dim = +dim.value; row.querySelector("output").textContent = `${Math.round(lamp.dim * 100)}%`; schedule(); };
  row.querySelector("input[type=number]").onchange = (e) => { lamp.hours = +e.target.value; schedule(); };
  row.querySelector(".del").onclick = () => { removeLamp(lamp); schedule(); };
  return row;
}
let glareTimer = 0;
function glareReport() {
  // Lamps in view, in plan coordinates (the group is mirrored in x).
  const toPlan = (v) => [W - v.x, v.y, v.z];
  const eye = toPlan(camera.position);
  const look = new THREE.Vector3(); camera.getWorldDirection(look);
  const up = camera.up.clone().applyQuaternion(camera.quaternion);
  const lk = [-look.x, look.y, look.z], u = [-up.x, up.y, up.z];
  const lamps = state.lamps.filter((l) => l.on && l.now.lumens > 0 && l.obj.area > 0).map((l) => {
    const d = l.pos.map((v, k) => eye[k] - v), r = Math.hypot(...d);
    const cos = l.fixture === "panel" ? Math.max(0, dot(d.map((v) => v / r), l.axis)) : 1;
    return { pos: l.pos, luminance: cos > 0 ? l.luminance : 0, area: l.obj.area * Math.max(cos, 0) };
  }).filter((l) => l.area > 0);
  // CIE background luminance: the indirect light reaching the eye, over π.
  const atEye = rad.luxAt(eye[0], eye[1], eye[2], ...lk);
  const Lb = Math.max(0.5, (atEye.bounced + atEye.sky) / Math.PI);
  const g = ugr(eye, lk, u, lamps, Lb);
  let text = `Glare from lamps (UGR): ${g == null ? "—" : fmt(g, 0)}, ${ugrWords(g)}.`;
  // Daylight: is the sun's disc, or a sunlit window, in view?
  if (out.sunLux > 0 && dot(lk, out.dir) > Math.cos(35 * DEG)) {
    const t = (WINDOW.z - eye[2]) / out.dir[2];
    const wx = eye[0] + out.dir[0] * t, wy = eye[1] + out.dir[1] * t;
    if (t > 0 && wx > WINDOW.x0 && wx < WINDOW.x1 && wy > WINDOW.y0 && wy < WINDOW.y1) text += " The sun itself is in view through the window: disabling glare.";
  }
  const eyeL = viewL || adaptL;
  if (out.winL > 25 * eyeL) text += ` The window is ${fmt(out.winL / eyeL, 0)}× brighter than the room around it: expect squinting against it.`;
  $("glare").textContent = text;
  $("adapt").textContent = `Eye adapted to ${fmt(eyeL, eyeL < 10 ? 1 : 0)} cd/m² in view; the room averages ${fmt(adaptL, adaptL < 10 ? 1 : 0)} cd/m² (exposure EV ${state.ev >= 0 ? "+" : ""}${state.ev}).`;
}
controls.addEventListener("change", () => { clearTimeout(glareTimer); glareTimer = setTimeout(() => out && glareReport(), 150); });

// ---------------------------------------------------------------- recommended levels
// EN 12464-1 / IES Lighting Handbook guidance for homes, in lux on the task.
function need(f) {
  if (f.owner === "desk" && f.top) return [500, "writing and reading at a desk"];
  if (f.owner === "bed" && f.top) return [300, "reading in bed"];
  if (f.name.startsWith("floor")) return [100, "general light in a bedroom"];
  return null;
}
let pickMode = null;     // null = measure; a fixture key = placing that lamp
const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
let downAt = null;
renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
  ptr.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(ptr, camera);
  const hit = ray.intersectObjects(meshes.filter((m) => m.visible), false)[0];
  if (!hit) return;
  const f = hit.object.userData.face, p = group.worldToLocal(hit.point.clone()).toArray();
  if (pickMode) { placeLamp(pickMode, f, p); pickMode = null; $("place").textContent = "Place"; document.body.classList.remove("placing"); return; }
  measure(f, p);
});
let marker = new THREE.Mesh(new THREE.SphereGeometry(0.025, 12, 8), new THREE.MeshBasicMaterial({ color: 0xff9f0a, toneMapped: false }));
marker.visible = false; group.add(marker);
let lastMeasure = null;
function measure(f, p) {
  lastMeasure = [f, p];
  marker.position.fromArray(p); marker.visible = true;
  const m = rad.luxAt(p[0], p[1], p[2], ...f.n);
  const total = m.lamps + m.sun + m.sky + m.bounced;
  const n = need(f);
  const rho = 0.2126 * f.rho[0] + 0.7152 * f.rho[1] + 0.0722 * f.rho[2];
  $("meter").innerHTML = `<b>${fmt(total, total < 10 ? 1 : 0)} lux</b> on ${f.name.replace(/-/g, " ")}
    <div class="fine">lamps ${fmt(m.lamps)} · sun ${fmt(m.sun)} · sky ${fmt(m.sky)} · bounced ${fmt(m.bounced)}.
    Looks ${fmt((rho * total) / Math.PI, 1)} cd/m² bright.${n ? ` Recommended for ${n[1]}: ${n[0]} lux — ${total >= n[0] ? "met" : `${fmt(100 * total / n[0])}% of it`}.` : ""}</div>
    ${materialControls(f)}`;
  const sel = $("meter").querySelector("select"), rough = $("meter").querySelector(".rough"), gain = $("meter").querySelector(".gain");
  sel.onchange = () => { f.kind = sel.value; f.lit.roughness = KINDS[f.kind].roughness; f.lit.sheen = KINDS[f.kind].sheen || 0; measure(f, p); };
  rough.oninput = () => { f.lit.roughness = +rough.value; };
  gain.onchange = () => { f.gain = +gain.value; setAlbedo(); schedule(); };
}
function materialControls(f) {
  const opts = Object.entries(KINDS).map(([k, v]) => `<option value="${k}" ${k === f.kind ? "selected" : ""}>${v.label}</option>`).join("");
  const rho = 0.2126 * f.rho[0] + 0.7152 * f.rho[1] + 0.0722 * f.rho[2];
  const seen = f.tex && coverage[f.tex] != null ? `${Math.round(coverage[f.tex] * 100)}% filmed` : "not filmed: plain colour";
  return `<div class="mat"><div class="fine">Material: reflects ${fmt(rho * 100)}% of light (${f.photo ? "from its photograph" : "assumed"}), ${seen}.</div>
    <label class="slider">Made of <select>${opts}</select></label>
    <label class="slider">Gloss <input class="rough" type="range" min="0.03" max="1" step="0.01" value="${f.lit.roughness}" style="direction: rtl"></label>
    <label class="slider">Lighter/darker <input class="gain" type="range" min="0.4" max="1.6" step="0.05" value="${f.gain}"></label></div>`;
}
function placeLamp(type, f, p) {
  let pos, axis = [0, -1, 0], kind = type;
  if (Math.abs(f.n[1]) < 0.5) {                     // a wall or the side of a piece: mounted on it
    pos = p.map((v, k) => v + f.n[k] * 0.06); axis = f.n.slice();
    if (type === "ceiling" || type === "table") kind = "tube";
  } else if (f.n[1] < -0.5) { pos = [p[0], p[1] - 0.03, p[2]]; if (type === "table") kind = "ceiling"; }
  else { pos = [p[0], p[1] + (f.name.startsWith("floor") ? 1.5 : 0.45), p[2]]; if (type === "ceiling" || type === "tube") kind = "table"; }
  addLamp(kind, pos, axis, { name: `${FIXTURES[kind].label} ${nextLamp}` });
  schedule();
}

// ---------------------------------------------------------------- panel wiring
let pending = false;
function schedule() { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; relight(); if (lastMeasure) measure(...lastMeasure); }); }
const bind = (id, key, conv = (v) => v, ev = "input") => {
  const el = $(id); el.value = state[key];
  el.addEventListener(ev, () => { state[key] = conv(el.value); if (id === "time") $("timeout").textContent = clock(); if (id === "ev") state.ev = +el.value; schedule(); });
};
const clock = () => `${String(Math.floor(state.minutes / 60)).padStart(2, "0")}:${String(state.minutes % 60).padStart(2, "0")}`;
bind("date", "date", String, "change"); bind("time", "minutes", Number); bind("lat", "lat", Number, "change"); bind("lon", "lon", Number, "change");
bind("utc", "utc", Number, "change"); bind("facing", "facing", Number, "change"); bind("sky", "sky", String, "change");
bind("curtain", "curtain", String, "change"); bind("volts", "volts", Number); bind("tariff", "tariff", Number, "change");
bind("currency", "currency", String, "change"); bind("co2", "co2", Number, "change"); bind("ev", "ev", Number);
$("timeout").textContent = clock();
$("voltsout").textContent = `${state.volts} V`;
$("volts").addEventListener("input", () => { $("voltsout").textContent = `${state.volts} V`; });
for (const [id, key] of [["delit", "delit"], ["heat", "heat"], ["glareon", "glare"]]) {
  const el = $(id); el.checked = state[key];
  el.onchange = () => { state[key] = el.checked; if (key === "delit") setAlbedo(); $("legend").hidden = !state.heat; schedule(); };
}
$("legend").hidden = !state.heat;
const fixtureSel = $("fixture");
for (const [k, v] of Object.entries(FIXTURES)) fixtureSel.add(new Option(v.label, k));
$("place").onclick = () => {
  pickMode = pickMode ? null : fixtureSel.value;
  $("place").textContent = pickMode ? "Click a surface…" : "Place";
  document.body.classList.toggle("placing", !!pickMode);
};
$("allon").onclick = () => { state.lamps.forEach((l) => (l.on = true)); schedule(); };
$("alloff").onclick = () => { state.lamps.forEach((l) => (l.on = false)); schedule(); };
$("toggle").onclick = () => document.body.classList.toggle("collapsed");

// Views (as in handbuilt-room/photo.js).
const VIEWS = {
  "Dollhouse": { from: [5.4, 4.6, 5.9], to: [W / 2, 0.9, L / 2], fov: 40, open: true },
  "From the desk": { from: [2.3, 1.45, 2.9], to: [1.0, 1.1, 0], fov: 69 },
  "From the door": { from: [2.4, 1.4, 0.35], to: [1.6, 1.0, L], fov: 69 },
  "Across the bed": { from: [2.3, 1.45, 1.0], to: [0, 1.15, 2.2], fov: 69 },
  "Sitting at the desk": { from: [1.8, 1.2, 2.55], to: [2.82, 0.95, 2.8], fov: 69 },
};
function setView(name) {
  const v = VIEWS[name] || VIEWS["Dollhouse"];
  camera.fov = v.fov; camera.position.set(W - v.from[0], v.from[1], v.from[2]);
  controls.target.set(W - v.to[0], v.to[1], v.to[2]);
  for (const m of ceilingMeshes) m.visible = !v.open;
  camera.updateProjectionMatrix(); controls.update();
  if (out) glareReport();
}
for (const name of Object.keys(VIEWS)) {
  const b = document.createElement("button"); b.textContent = name; b.onclick = () => setView(name); $("views").appendChild(b);
}
if (params.get("time")) { const [h, m] = params.get("time").split(":").map(Number); state.minutes = h * 60 + (m || 0); $("time").value = state.minutes; $("timeout").textContent = clock(); }
if (params.get("date")) { state.date = params.get("date"); $("date").value = state.date; }
if (params.get("curtain")) { state.curtain = params.get("curtain"); $("curtain").value = state.curtain; }
if (params.get("lamps") === "off") state.lamps.forEach((l) => (l.on = false));
if (params.has("heat")) { state.heat = true; $("heat").checked = true; $("legend").hidden = false; }
if (params.has("filmed")) { state.delit = false; $("delit").checked = false; setAlbedo(); }
if (params.has("shot")) document.body.classList.add("collapsed", "shot");
setView(params.get("view") || "Dollhouse");
relight();

addEventListener("resize", () => {
  renderer.setSize(innerWidth, innerHeight); composer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
});
// The eye adapts to what is in view: a small HDR render of the view, its log-average luminance
// (the brightest 2% left out, so a bulb or the sun's patch does not darken everything), and an
// exposure that follows it over about half a second. Not fully in the dark: below ~1.5 cd/m²
// a room looks dim, as it does to a person.
const meterRT = new THREE.WebGLRenderTarget(64, 36, { type: THREE.FloatType });
const meterBuf = new Float32Array(64 * 36 * 4);
let frame = 0, lastT = performance.now();
function meter() {
  const bg = scene.background; scene.background = null;          // the void around the room does not count
  const clear = renderer.getClearAlpha(); renderer.setClearAlpha(0);
  renderer.setRenderTarget(meterRT); renderer.clear(); renderer.render(scene, camera); renderer.setRenderTarget(null);
  scene.background = bg; renderer.setClearAlpha(clear);
  renderer.readRenderTargetPixels(meterRT, 0, 0, 64, 36, meterBuf);
  const logs = [];
  for (let i = 0; i < 64 * 36; i++) {
    const y = 0.2126 * meterBuf[i * 4] + 0.7152 * meterBuf[i * 4 + 1] + 0.0722 * meterBuf[i * 4 + 2];
    if (meterBuf[i * 4 + 3] > 0.5) logs.push(Math.log(y + 1e-3));
  }
  if (!logs.length) return null;
  logs.sort((a, b) => a - b);
  const keep = logs.slice(0, Math.max(1, Math.floor(logs.length * 0.98)));
  return Math.exp(keep.reduce((a, b) => a + b, 0) / keep.length);
}
renderer.setAnimationLoop(() => {
  controls.update();
  const now = performance.now(), dt = Math.min(0.5, (now - lastT) / 1000); lastT = now;
  if (frame++ % 10 === 0 || metered < 3) {
    const L = meter();
    if (L != null) {
      const target = Math.max(L, 1.5);
      viewL = viewL == null || metered < 3 ? target : Math.exp(Math.log(viewL) + (Math.log(target) - Math.log(viewL)) * Math.min(1, dt * 10 / 0.5));
      metered++;
    }
  }
  if (state.heat) { exposurePass.uniforms.exposure.value = 1; bloom.strength = 0; }
  else if (viewL) {
    bloom.strength = state.glare ? 0.2 : 0;
    const key = 1.03 - 2 / (2 + Math.log10(viewL + 1));
    exposure = (key / viewL) * Math.pow(2, state.ev);
    exposurePass.uniforms.exposure.value = exposure;
  }
  composer.render();
});
window.labReady = true;
window.lab = { state, relight, rad, faces, setView, measure, out: () => out };
