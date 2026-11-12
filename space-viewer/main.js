// Space viewer: a space straight after stage 3 (tools/space_viewer_export.py writes
// what this reads), shown as its dense cloud with the room model over it. Take a
// piece out (its points go, the floor under it is filled and marked as inferred),
// put furniture in from Poly Haven (CC0), a link or a file, move, turn and size it.
// The layout is kept in this browser and can be exported.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const params = new URLSearchParams(location.search);
const SPACE = params.get("space") || "bedroom";
const BASE = new URL(params.get("data") || `../spaces/${SPACE}/viewer/`, location.href);
const STORE = `space-viewer:${SPACE}`;
const MAX_PIECES = 63;          // the shader's hide list
const POLY = "https://api.polyhaven.com";

const $ = (id) => document.getElementById(id);
const say = (text) => { $("status").textContent = text; };

// ------------------------------------------------------------------ scene
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x121417);
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.02, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(2, 4, 3);
scene.add(sun);

const modelGroup = new THREE.Group();     // walls, floor outline, boxes
const pathGroup = new THREE.Group();
const fillGroup = new THREE.Group();      // floor filled where a piece was taken out
const addedGroup = new THREE.Group();
const measureGroup = new THREE.Group();   // measurement lines
// Everything measured from the video lives in spaceRoot, whose scale a real length sets
// (Set real length); added furniture stays at its true size outside it.
const spaceRoot = new THREE.Group();
spaceRoot.add(modelGroup, pathGroup, fillGroup, measureGroup);
scene.add(spaceRoot, addedGroup);
pathGroup.visible = false;

let room = null;
let points = null;
const hidden = new Float32Array(MAX_PIECES + 1);
const GUESSED = 255;           // the piece index of points that fill a gap (inferred, not measured)
const pieces = new Map();      // id -> {info, box (helper), pick (invisible mesh), fill, removed}
const added = new Map();       // id -> {group, source, name}
let selected = null;
let nextAdded = 1;
let factor = 1;                // real metres per reconstructed metre (1 until a real length is set)
let realScale = null;          // {a, b, real}: the measurement that set it
const measures = [];           // {a, b (Vector3, reconstructed metres), line, label}
let measuring = null;          // null, or {first: Vector3 | null}

function resize() {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  if (points) points.material.uniforms.scale.value = innerHeight / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
}
addEventListener("resize", resize);

// ------------------------------------------------------------------ the cloud
function pointsMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: { size: { value: 0.012 }, scale: { value: 1 }, hidden: { value: hidden },
                showGuess: { value: 1 }, markGuess: { value: 0 } },
    vertexShader: `
      attribute vec3 color; attribute float piece;
      uniform float size; uniform float scale; uniform float hidden[${MAX_PIECES + 1}];
      uniform float showGuess; uniform float markGuess;
      varying vec3 vColor; varying float vHide;
      void main() {
        vColor = color;
        int p = int(piece + 0.5);
        vHide = (p > 0 && p <= ${MAX_PIECES}) ? hidden[p] : 0.0;
        // ${GUESSED}: a gap in a wall, the floor or the ceiling filled from what surrounds it.
        if (p == ${GUESSED}) {
          vHide = showGuess > 0.5 ? 0.0 : 1.0;
          if (markGuess > 0.5) vColor = mix(color, vec3(1.0, 0.54, 0.24), 0.6);
        }
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = max(1.0, size * scale / -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3 vColor; varying float vHide;
      void main() { if (vHide > 0.5) discard; gl_FragColor = vec4(vColor, 1.0); }`,
  });
}

async function loadSpace() {
  room = await (await fetch(new URL("room.json", BASE))).json();
  const buffer = await (await fetch(new URL("points.bin", BASE))).arrayBuffer();
  const n = room.count;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(buffer, 0, n * 3), 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(new Uint8Array(buffer, n * 12, n * 3), 3, true));
  geometry.setAttribute("piece", new THREE.BufferAttribute(new Uint8Array(buffer, n * 15, n), 1));
  points = new THREE.Points(geometry, pointsMaterial());
  points.frustumCulled = false;
  spaceRoot.add(points);
  resize();
  buildModel();
  $("title").textContent = room.space.replace(/-/g, " ");
  const span = bounds();
  updateSubtitle();
  viewFromAbove();
}

function bounds() {
  // In real metres: the reconstructed room times the scale a real length set.
  const xs = room.floor.flat().map((p) => p[0] * factor), zs = room.floor.flat().map((p) => p[1] * factor);
  return { x: Math.max(...xs) - Math.min(...xs), z: Math.max(...zs) - Math.min(...zs),
           cx: (Math.max(...xs) + Math.min(...xs)) / 2, cz: (Math.max(...zs) + Math.min(...zs)) / 2 };
}

function updateSubtitle() {
  const span = bounds();
  $("subtitle").textContent = `${span.x.toFixed(2)} × ${span.z.toFixed(2)} m, ${(room.height * factor).toFixed(2)} m tall · ` +
    `${room.count.toLocaleString()} points · ${room.pieces.length} pieces`;
}

// ------------------------------------------------------------------ the room model
function buildModel() {
  const wallMat = new THREE.MeshBasicMaterial({ color: 0xff6b6b, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false });
  const guessMat = new THREE.MeshBasicMaterial({ color: 0xffb35c, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false });
  for (const w of room.walls) {
    const [x0, z0] = w.from, [x1, z1] = w.to;
    const length = Math.hypot(x1 - x0, z1 - z0);
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(length, room.height), w.inferred ? guessMat : wallMat);
    plane.position.set((x0 + x1) / 2, room.height / 2, (z0 + z1) / 2);
    plane.rotation.y = -Math.atan2(z1 - z0, x1 - x0);
    modelGroup.add(plane);
    const edge = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x0, 0.01, z0), new THREE.Vector3(x1, 0.01, z1)]),
                                new THREE.LineBasicMaterial({ color: w.inferred ? 0xffb35c : 0xff6b6b }));
    modelGroup.add(edge);
  }
  for (const poly of room.floor) {
    const pts = poly.map(([x, z]) => new THREE.Vector3(x, 0.005, z));
    modelGroup.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x6fb3ff })));
  }
  for (const info of room.pieces.slice(0, MAX_PIECES)) {
    const size = info.max.map((v, k) => v - info.min[k]);
    const centre = info.max.map((v, k) => (v + info.min[k]) / 2);
    const geo = new THREE.BoxGeometry(...size);
    const box = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0x5ed38a }));
    box.position.set(...centre);
    modelGroup.add(box);
    const pick = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ visible: false }));
    pick.position.set(...centre);
    pick.userData = { kind: "piece", id: info.id };
    spaceRoot.add(pick);
    pieces.set(info.id, { info, box, pick, fill: null, removed: false });
  }
  if (room.path.length > 1) {
    const pts = room.path.map(([x, z]) => new THREE.Vector3(x, 1.4, z));
    pathGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0xc77dff })));
  }
}

// ------------------------------------------------------------------ taking pieces out
function floorFill(info) {
  // The floor under a piece was never filmed: fill it with the room's floor colour,
  // marked as inferred (highlighted with "Highlight filled-in floor").
  const [r, g, b] = room.floorColor;
  const w = info.max[0] - info.min[0], d = info.max[2] - info.min[2];
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, d),
    new THREE.MeshStandardMaterial({ color: new THREE.Color(r / 255, g / 255, b / 255).convertSRGBToLinear(), roughness: 0.9 }));
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.set((info.max[0] + info.min[0]) / 2, 0.004, (info.max[2] + info.min[2]) / 2);
  mesh.userData.baseColor = mesh.material.color.clone();
  return mesh;
}

function setRemoved(id, removed, save = true) {
  const p = pieces.get(id);
  if (!p) return;
  p.removed = removed;
  hidden[p.info.index] = removed ? 1 : 0;
  p.box.visible = !removed;
  p.pick.visible = !removed;
  if (removed && !p.fill && !p.info.mounted && p.info.min[1] < 0.15) {
    p.fill = floorFill(p.info);
    fillGroup.add(p.fill);
  } else if (!removed && p.fill) {
    fillGroup.remove(p.fill);
    p.fill = null;
  }
  highlightFilled($("show-filled").checked);
  if (save) persist();
  renderList();
  if (selected === id) showSelected();
  say(removed ? `Took out the ${p.info.name}; the floor under it is filled in (inferred).` : `Put the ${p.info.name} back.`);
}

function highlightFilled(on) {
  for (const mesh of fillGroup.children) mesh.material.color.copy(on ? new THREE.Color(0xff8a3d) : mesh.userData.baseColor);
  if (points) points.material.uniforms.markGuess.value = on ? 1 : 0;
}

// ------------------------------------------------------------------ adding furniture
const loader = new GLTFLoader();

function prepare(object) {
  // Stand the model on the floor, centred on its footprint, inside a group we move.
  const box = new THREE.Box3().setFromObject(object);
  const centre = box.getCenter(new THREE.Vector3());
  object.position.sub(new THREE.Vector3(centre.x, box.min.y, centre.z));
  const group = new THREE.Group();
  group.add(object);
  return group;
}

async function addModel(source, place = null, save = true) {
  say(`Loading ${source.name}…`);
  let gltf;
  if (source.kind === "polyhaven") {
    const files = await (await fetch(`${POLY}/files/${source.id}`)).json();
    const res = files.gltf["1k"] || Object.values(files.gltf)[0];
    gltf = await loader.loadAsync(res.gltf.url);
  } else if (source.kind === "url") {
    gltf = await loader.loadAsync(source.url);
  } else {
    gltf = await loader.parseAsync(source.data, "");
  }
  const group = prepare(gltf.scene);
  const at = place || { x: bounds().cx, z: bounds().cz, rot: 0, scale: 1 };
  group.position.set(at.x, 0, at.z);
  group.rotation.y = at.rot || 0;
  group.scale.setScalar(at.scale || 1);
  const id = `A${nextAdded++}`;
  group.userData = { kind: "added", id };
  group.traverse((o) => { o.userData.addedId = id; });
  addedGroup.add(group);
  added.set(id, { group, source: source.kind === "file" ? { kind: "file", name: source.name } : source, name: source.name });
  if (save) persist();
  renderList();
  select(id);
  const size = new THREE.Box3().setFromObject(group).getSize(new THREE.Vector3());
  say(`Added ${source.name} (${size.x.toFixed(2)} × ${size.z.toFixed(2)} m, ${size.y.toFixed(2)} m tall). Drag it to move it.`);
  return id;
}

function removeAdded(id) {
  const a = added.get(id);
  if (!a) return;
  addedGroup.remove(a.group);
  added.delete(id);
  if (selected === id) select(null);
  persist();
  renderList();
}

// ------------------------------------------------------------------ the catalogue
let catalogue = null;
async function openCatalogue() {
  $("catalogue").classList.add("show");
  if (!catalogue) {
    $("grid").textContent = "Loading Poly Haven's furniture…";
    try {
      const list = await (await fetch(`${POLY}/assets?t=models&c=furniture`)).json();
      catalogue = Object.entries(list).map(([id, a]) => ({ id, name: a.name || id, tags: [...(a.categories || []), ...(a.tags || [])] }));
    } catch (err) {
      $("grid").textContent = `Could not reach Poly Haven (${err.message}). Paste a model link instead.`;
      return;
    }
  }
  renderCatalogue();
}

function renderCatalogue() {
  const q = $("search").value.trim().toLowerCase();
  const grid = $("grid");
  grid.textContent = "";
  for (const item of catalogue.filter((c) => !q || c.name.toLowerCase().includes(q) || c.tags.some((t) => t.includes(q)))) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<img loading="lazy" alt="" src="https://cdn.polyhaven.com/asset_img/thumbs/${item.id}.png?width=256&height=256"><div></div>`;
    card.querySelector("div").textContent = item.name;
    card.onclick = () => {
      $("catalogue").classList.remove("show");
      addModel({ kind: "polyhaven", id: item.id, name: item.name }).catch((err) => say(`Could not load ${item.name}: ${err.message}`));
    };
    grid.appendChild(card);
  }
}

// ------------------------------------------------------------------ the panel
function renderList() {
  const list = $("pieces");
  list.textContent = "";
  const row = (id, name, meta, tag, cls) => {
    const el = document.createElement("div");
    el.className = `piece ${cls || ""} ${selected === id ? "selected" : ""}`;
    el.innerHTML = `<div><div class="name"></div><div class="meta"></div></div><span class="tag"></span>`;
    el.querySelector(".name").textContent = name;
    el.querySelector(".meta").textContent = meta;
    const t = el.querySelector(".tag");
    t.textContent = tag;
    if (tag === "added") t.classList.add("added");
    el.onclick = () => select(id);
    list.appendChild(el);
  };
  for (const [id, p] of pieces) {
    const s = p.info.max.map((v, k) => (v - p.info.min[k]) * factor);
    row(id, p.info.name, `${s[0].toFixed(2)} × ${s[2].toFixed(2)} m, ${s[1].toFixed(2)} m tall`, p.removed ? "taken out" : "filmed", p.removed ? "removed" : "");
  }
  for (const [id, a] of added) row(id, a.name, "drag to move", "added");
}

function select(id) {
  selected = id;
  renderList();
  showSelected();
  for (const [pid, p] of pieces) p.box.material.color.set(pid === id ? 0x6fb3ff : 0x5ed38a);
}

function showSelected() {
  const panel = $("selected");
  const actions = $("sel-actions");
  actions.textContent = "";
  $("sel-scale").style.display = "none";
  if (!selected) { panel.classList.remove("show"); return; }
  panel.classList.add("show");
  const button = (text, fn, cls = "") => { const b = document.createElement("button"); b.textContent = text; b.className = cls; b.onclick = fn; actions.appendChild(b); };
  if (pieces.has(selected)) {
    const p = pieces.get(selected);
    $("sel-name").textContent = p.info.name;
    $("sel-meta").textContent = `Measured from the video (${p.info.points.toLocaleString()} points)` + (p.info.mounted ? ", on the wall" : "");
    if (p.removed) button("Put it back", () => setRemoved(selected, false));
    else button("Take it out", () => setRemoved(selected, true), "danger");
  } else if (added.has(selected)) {
    const a = added.get(selected);
    $("sel-name").textContent = a.name;
    $("sel-meta").textContent = "Added · drag it on the floor to move it";
    button("⟲ 15°", () => turn(15));
    button("⟳ 15°", () => turn(-15));
    button("Remove", () => removeAdded(selected), "danger");
    $("sel-scale").style.display = "block";
    const pct = Math.round(a.group.scale.x * 100);
    $("scale").value = pct;
    $("sel-scale-value").textContent = `${pct}%`;
  }
}

function turn(degrees) {
  const a = added.get(selected);
  if (!a) return;
  a.group.rotation.y += THREE.MathUtils.degToRad(degrees);
  persist();
}

// ------------------------------------------------------------------ picking and dragging
const ray = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let dragging = null, downAt = null;

function aim(event) {
  const r = renderer.domElement.getBoundingClientRect();
  pointer.set(((event.clientX - r.left) / r.width) * 2 - 1, -((event.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(pointer, camera);
}

renderer.domElement.addEventListener("pointerdown", (event) => {
  downAt = { x: event.clientX, y: event.clientY };
  if (measuring) return;
  aim(event);
  const hit = ray.intersectObjects(addedGroup.children, true)[0];
  if (hit) {
    const id = hit.object.userData.addedId;
    select(id);
    const at = ray.ray.intersectPlane(floorPlane, new THREE.Vector3());
    if (at) {
      const g = added.get(id).group;
      dragging = { id, offset: new THREE.Vector3().subVectors(g.position, at) };
      controls.enabled = false;
    }
  }
});
renderer.domElement.addEventListener("pointermove", (event) => {
  if (!dragging) return;
  aim(event);
  const at = ray.ray.intersectPlane(floorPlane, new THREE.Vector3());
  if (at) added.get(dragging.id).group.position.set(at.x + dragging.offset.x, 0, at.z + dragging.offset.z);
});
renderer.domElement.addEventListener("pointerup", (event) => {
  if (dragging) { dragging = null; controls.enabled = true; persist(); return; }
  if (!downAt || Math.hypot(event.clientX - downAt.x, event.clientY - downAt.y) > 5) return;   // a drag of the view
  aim(event);
  if (measuring) { measureClick(); return; }
  const picks = [...pieces.values()].filter((p) => !p.removed).map((p) => p.pick);
  const hit = ray.intersectObjects(picks, false)[0];
  select(hit ? hit.object.userData.id : null);
});

// ------------------------------------------------------------------ measuring
function surfaceUnderPointer() {
  // The nearest visible point of the cloud under the pointer, else a box's face, else
  // the floor; in reconstructed metres (spaceRoot's own frame).
  ray.params.Points.threshold = 0.02;
  const piece = points.geometry.getAttribute("piece");
  const hit = ray.intersectObject(points, false).find((h) => !hidden[piece.getX(h.index)]);
  const box = ray.intersectObjects([...pieces.values()].filter((p) => !p.removed).map((p) => p.pick), false)[0];
  let at = hit ? hit.point : null;
  if (box && (!at || box.distance < hit.distance - 0.05)) at = box.point;
  if (!at) at = ray.ray.intersectPlane(floorPlane, new THREE.Vector3());
  return at ? spaceRoot.worldToLocal(at.clone()) : null;
}

function startMeasuring() {
  measuring = { first: null };
  $("measure").textContent = "Cancel measuring";
  say("Click the first point of the length you want to measure.");
}
function stopMeasuring(text) {
  measuring = null;
  $("measure").textContent = "Measure a distance";
  if (text) say(text);
}

function measureClick() {
  const at = surfaceUnderPointer();
  if (!at) return;
  if (!measuring.first) {
    measuring.first = at;
    say("Now click the second point.");
    return;
  }
  const m = addMeasure(measuring.first, at);
  stopMeasuring(`${fmt(lengthOf(m))} between the two points. Know the real length? Set it in the panel.`);
}

const lengthOf = (m) => m.a.distanceTo(m.b) * factor;
const fmt = (metres) => (metres < 1 ? `${(metres * 100).toFixed(1)} cm` : `${metres.toFixed(3)} m`);

function addMeasure(a, b, save = true) {
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]),
                              new THREE.LineBasicMaterial({ color: 0xffd166, depthTest: false }));
  line.renderOrder = 10;
  const dots = new THREE.Points(new THREE.BufferGeometry().setFromPoints([a, b]),
                                new THREE.PointsMaterial({ color: 0xffd166, size: 8, sizeAttenuation: false, depthTest: false }));
  dots.renderOrder = 10;
  line.add(dots);
  measureGroup.add(line);
  const label = document.createElement("div");
  label.className = "measure-label";
  $("labels").appendChild(label);
  const m = { a: a.clone(), b: b.clone(), line, label };
  measures.push(m);
  renderMeasures();
  if (save) persist();
  return m;
}

function removeMeasure(m) {
  measureGroup.remove(m.line);
  m.label.remove();
  measures.splice(measures.indexOf(m), 1);
  if (realScale && realScale.measure === m) setScale(1, null);
  renderMeasures();
  persist();
}

function setScale(value, from, save = true) {
  // `value` real metres per reconstructed metre; `from` the measurement and the real length behind it.
  factor = value;
  realScale = from;
  spaceRoot.scale.setScalar(factor);
  updateSubtitle();
  renderList();
  renderMeasures();
  if (save) persist();
}

function setRealLength(m, metres) {
  const measured = m.a.distanceTo(m.b);
  if (!(metres > 0) || measured <= 0) return;
  setScale(metres / measured, { measure: m, real: metres });
  const change = (factor - 1) * 100;
  say(`Scale set from a ${fmt(metres)} length: the space is ${Math.abs(change).toFixed(1)}% ` +
      `${change >= 0 ? "larger" : "smaller"} than reconstructed.`);
}

function renderMeasures() {
  const list = $("measures");
  list.textContent = "";
  for (const m of measures) {
    const row = document.createElement("div");
    row.className = "piece";
    const isScale = realScale && realScale.measure === m;
    row.innerHTML = `<div><div class="name"></div><div class="meta"></div></div><div class="row"></div>`;
    row.querySelector(".name").textContent = fmt(lengthOf(m));
    row.querySelector(".meta").textContent = isScale ? "sets the real scale" : `reconstructed ${fmt(m.a.distanceTo(m.b))}`;
    const buttons = row.querySelector(".row");
    const set = document.createElement("button");
    set.textContent = "Set real length";
    set.onclick = (e) => {
      e.stopPropagation();
      const answer = prompt("The real length, in metres (e.g. 0.90 for a 90 cm door):", lengthOf(m).toFixed(3));
      if (answer !== null) setRealLength(m, parseFloat(answer.replace(",", ".")));
    };
    const del = document.createElement("button");
    del.textContent = "×";
    del.className = "danger";
    del.onclick = (e) => { e.stopPropagation(); removeMeasure(m); };
    buttons.append(set, del);
    list.appendChild(row);
    m.label.textContent = fmt(lengthOf(m));
  }
  const change = (factor - 1) * 100;
  const fixed = room.scaleMeasured;
  $("scale-status").textContent = realScale
    ? `Scale set from a real ${fmt(realScale.real)}: ${Math.abs(change).toFixed(1)}% ${change >= 0 ? "larger" : "smaller"} than reconstructed. ` +
      "Export the layout and run tools/set_real_scale.py to fix it in the pipeline."
    : fixed ? `Real scale, fixed in the pipeline from a measured ${fmt(fixed.real_m)}.`
            : "Scale as reconstructed (from the video alone).";
  $("reset-scale").disabled = !realScale;
}

function placeLabels() {
  const r = renderer.domElement.getBoundingClientRect();
  for (const m of measures) {
    const mid = spaceRoot.localToWorld(m.a.clone().add(m.b).multiplyScalar(0.5)).project(camera);
    const visible = mid.z < 1 && measureGroup.visible;
    m.label.style.display = visible ? "block" : "none";
    m.label.style.transform = `translate(${(mid.x + 1) / 2 * r.width}px, ${(1 - mid.y) / 2 * r.height}px) translate(-50%, -50%)`;
  }
}

// ------------------------------------------------------------------ views
function viewFromAbove() {
  const b = bounds();
  controls.target.set(b.cx, 0, b.cz);
  camera.position.set(b.cx, Math.max(b.x, b.z) * 1.35 + room.height, b.cz + 0.01);
  controls.update();
}
function viewFromInside() {
  const p = room.path[Math.floor(room.path.length / 2)] || [0, 0];
  const b = bounds();
  camera.position.set(p[0], 1.5, p[1]);
  controls.target.set(b.cx + (b.cx - p[0]) * 0.3, 1.0, b.cz + (b.cz - p[1]) * 0.3);
  controls.update();
}

// ------------------------------------------------------------------ keeping the layout
function layout() {
  return {
    space: SPACE,
    scaleSource: room.scaleSource || "video",
    removed: [...pieces].filter(([, p]) => p.removed).map(([id]) => id),
    measures: measures.map((m) => ({ a: m.a.toArray().map((v) => +v.toFixed(4)), b: m.b.toArray().map((v) => +v.toFixed(4)),
                                     metres: +lengthOf(m).toFixed(4) })),
    scale: realScale ? { factor: +factor.toFixed(5), real_m: realScale.real, measure: measures.indexOf(realScale.measure) } : null,
    added: [...added.values()].filter((a) => a.source.kind !== "file").map((a) => ({
      source: a.source, name: a.name, x: +a.group.position.x.toFixed(3), z: +a.group.position.z.toFixed(3),
      rot: +a.group.rotation.y.toFixed(4), scale: +a.group.scale.x.toFixed(3) })),
  };
}
function persist() {
  try { localStorage.setItem(STORE, JSON.stringify(layout())); } catch { /* private mode: not kept */ }
}
async function restore() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(STORE) || "null"); } catch { saved = null; }
  if (!saved) return;
  for (const id of saved.removed || []) setRemoved(id, true, false);
  // Measurements are kept in the metres of the export they were taken on; after the
  // space's scale is fixed in the pipeline (set_real_scale.py) they no longer fit it.
  if ((saved.scaleSource || "video") === (room.scaleSource || "video")) {
    for (const m of saved.measures || []) addMeasure(new THREE.Vector3(...m.a), new THREE.Vector3(...m.b), false);
    if (saved.scale && measures[saved.scale.measure]) setRealLength(measures[saved.scale.measure], saved.scale.real_m);
  } else if ((saved.measures || []).length) {
    say("The space now has its real scale from the pipeline; earlier measurements were cleared.");
  }
  for (const item of saved.added || []) {
    try { await addModel(item.source, item, false); } catch (err) { say(`Could not reload ${item.name}: ${err.message}`); }
  }
  select(null);
}

// ------------------------------------------------------------------ wiring
$("show-model").onchange = (e) => { modelGroup.visible = e.target.checked; };
$("show-path").onchange = (e) => { pathGroup.visible = e.target.checked; };
$("show-filled").onchange = (e) => highlightFilled(e.target.checked);
$("show-guess").onchange = (e) => { if (points) points.material.uniforms.showGuess.value = e.target.checked ? 1 : 0; };
$("point-size").oninput = (e) => { points.material.uniforms.size.value = e.target.value / 1000; };
$("scale").oninput = (e) => {
  const a = added.get(selected);
  if (!a) return;
  a.group.scale.setScalar(e.target.value / 100);
  $("sel-scale-value").textContent = `${e.target.value}%`;
  persist();
};
$("open-catalogue").onclick = () => openCatalogue();
$("close-catalogue").onclick = () => $("catalogue").classList.remove("show");
$("search").oninput = () => catalogue && renderCatalogue();
$("upload-button").onclick = () => $("upload").click();
$("upload").onchange = async (e) => {
  const file = e.target.files[0];
  if (file) addModel({ kind: "file", name: file.name, data: await file.arrayBuffer() }).catch((err) => say(`Could not read ${file.name}: ${err.message}`));
};
$("add-url").onclick = () => {
  const url = $("model-url").value.trim();
  if (url) addModel({ kind: "url", url, name: url.split("/").pop() }).catch((err) => say(`Could not load that link: ${err.message}`));
};
$("view-top").onclick = viewFromAbove;
$("measure").onclick = () => (measuring ? stopMeasuring("Measuring cancelled.") : startMeasuring());
$("reset-scale").onclick = () => { setScale(1, null); say("Scale back to the reconstruction's own."); };
$("view-inside").onclick = viewFromInside;
$("reset").onclick = () => {
  for (const id of [...added.keys()]) removeAdded(id);
  for (const id of pieces.keys()) setRemoved(id, false, false);
  for (const m of [...measures]) removeMeasure(m);
  setScale(1, null, false);
  persist();
  say("Back to the space as filmed.");
};
$("export").onclick = () => {
  const blob = new Blob([JSON.stringify(layout(), null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${SPACE}-layout.json`;
  a.click();
};

renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); placeLabels(); });

loadSpace()
  .then(restore)
  .then(() => { renderList(); say("Click a piece to take it out, or add furniture from the panel."); })
  .catch((err) => say(`Could not load the space: ${err.message}. Run tools/space_viewer_export.py ${SPACE} first.`));

window.spaceViewer = { pieces, added, setRemoved, addModel, removeAdded, select, layout, camera, scene, hidden,
                       measures, addMeasure, setRealLength, setScale, bounds, get factor() { return factor; } };
