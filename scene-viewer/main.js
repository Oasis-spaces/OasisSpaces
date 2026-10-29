// The mixed-scene viewer: the room's shell is a textured mesh (shell.glb), each
// piece of furniture is its own splat (pieces/*.splat), both drawn together by
// three.js and Spark. Pieces are selected by their measured boxes, dragged
// across the floor, turned about the vertical, hidden and put back, and given a
// colour of their own. Any GLB can be added to the room as a piece like the
// others, and a saved layout (where everything is, its colour, the added
// objects themselves) opens again with "Load layout" or ?layout=<url>.
// Everything is in metres, y up, the room's centre on the floor at the origin
// (see tools/mixed_scene.py, which writes scene.json).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

const params = new URLSearchParams(location.search);
const sceneURL = new URL(params.get("scene") || "../spaces/walkthrough-full/scene/scene.json", location.href);
const status = document.getElementById("status");
const toast = document.getElementById("toast");
let toastTimer = 0;
function say(text) {
  toast.textContent = text;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 4000);
}

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

// id -> {info, group, proxy, row, size, splat, model, generated, looks, showing,
//        home: {position, rotation}, tint: "#rrggbb" or null, glb: an added object's file as a data: URL}
const pieces = new Map();
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
  if (params.get("layout")) {
    const url = new URL(params.get("layout"), location.href);
    await applyLayout(await (await fetch(url)).json()).catch((error) => say(`Could not open the layout: ${error.message}`));
  }
}

// `object`, for a piece added from a GLB, is that object (its feet at the origin);
// pieces of the scan have a splat and may have a clean model beside it.
function addPiece(info, object = null) {
  const group = new THREE.Group();
  group.position.fromArray(info.anchor);
  const splat = info.file ? new SplatMesh({ url: new URL(info.file, sceneURL).href }) : null;
  if (splat) group.add(splat);
  // The clean stand-in beside the scan, and a generated model where one fitted
  // (pipeline/model_fit.py): one of them shows at a time.
  const model = new THREE.Group();
  const loads = [];
  if (object) {
    model.add(object);
    group.add(model);
  } else if (info.model) {
    loads.push(new GLTFLoader().loadAsync(new URL(info.model, sceneURL).href).then((gltf) => {
      // A photographed side (a wardrobe's doors, a table's top) holds the room's own light,
      // like the shell: drawn as it is, while the plain parts of the model take the lights.
      gltf.scene.traverse((node) => {
        if (node.isMesh && node.material.map) {
          const map = node.material.map;
          map.anisotropy = renderer.capabilities.getMaxAnisotropy();
          node.material = new THREE.MeshBasicMaterial({ map, side: THREE.FrontSide });
        }
      });
      model.add(gltf.scene);
    }));
    group.add(model);
  }
  let generated = null;
  if (info.generated) {
    // offset + turn(scale * (vertex - origin)), in the piece's frame: as model_fit.py placed it.
    const g = info.generated;
    generated = new THREE.Group();
    generated.position.fromArray(g.offset);
    generated.rotation.y = THREE.MathUtils.degToRad(g.turnDegrees);
    generated.scale.setScalar(g.scale);
    loads.push(new GLTFLoader().loadAsync(new URL(g.file, sceneURL).href).then((gltf) => {
      gltf.scene.position.set(-g.origin[0], -g.origin[1], -g.origin[2]);
      generated.add(gltf.scene);
    }));
    group.add(generated);
  }
  const modelReady = Promise.all(loads);
  const looks = Object.fromEntries(Object.entries({ scan: splat, model: object || info.model ? model : null, generated })
    .filter(([, look]) => look));
  const showing = looks[info.show] ? info.show : object ? "model" : "scan";
  for (const [kind, look] of Object.entries(looks)) look.visible = kind === showing;
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
  row.innerHTML = `<input type="checkbox" checked><span class="swatch"></span><span class="name"></span><span class="count"></span>`;
  row.querySelector(".name").textContent = info.label;
  row.querySelector(".count").textContent = info.imported ? "added" : info.movable ? showing : "";
  const box = row.querySelector("input");
  box.addEventListener("change", () => { group.visible = box.checked; if (!box.checked && selected === info.id) select(null); });
  row.addEventListener("click", (event) => { if (event.target !== box && info.movable && group.visible) { event.preventDefault(); select(info.id); } });
  document.getElementById(info.imported ? "added" : info.movable ? "furniture" : "fixed").append(row);
  const piece = { info, group, proxy, row, size, splat, model, generated, looks, showing, modelReady, tint: null };
  pieces.set(info.id, piece);
  return piece;
}

// ------------------------------------------------------------ added objects
let added = 0;
async function addObject(data, name, at = null) {
  const gltf = await new GLTFLoader().parseAsync(data, "");
  const object = gltf.scene;
  let bounds = new THREE.Box3().setFromObject(object);
  if (bounds.isEmpty()) throw new Error("it has no meshes");
  // glTF is in metres; a model tens of metres across was made in centimetres.
  const across = Math.max(...bounds.getSize(new THREE.Vector3()).toArray());
  if (across > 20) {
    object.scale.multiplyScalar(0.01);
    bounds = new THREE.Box3().setFromObject(object);
    say(`${name} was ${across.toFixed(0)} units across: read as centimetres`);
  }
  const centre = bounds.getCenter(new THREE.Vector3());
  object.position.sub(new THREE.Vector3(centre.x, bounds.min.y, centre.z));   // its feet at the origin
  bounds = new THREE.Box3().setFromObject(object);
  // On the floor where the view is aimed, inside the room.
  const r = manifest.room, target = at ?? controls.target;
  const anchor = [THREE.MathUtils.clamp(target.x, -r.width / 2, r.width / 2), 0,
                  THREE.MathUtils.clamp(target.z, -r.depth / 2, r.depth / 2)];
  let id;
  do { id = `added-${++added}`; } while (pieces.has(id));
  const info = { id, label: name.replace(/\.(glb|gltf)$/i, "").replace(/[-_]+/g, " "), movable: true, imported: true,
                 anchor, box: { min: bounds.min.toArray(), max: bounds.max.toArray() } };
  const piece = addPiece(info, object);
  piece.glb = await dataURL(data, "model/gltf-binary");
  piece.home = { position: piece.group.position.clone(), rotation: 0 };
  return piece;
}

function dataURL(data, type) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(new Blob([data], { type }));
  });
}

async function addFiles(files) {
  for (const file of files) {
    try {
      if (/\.json$/i.test(file.name)) await applyLayout(JSON.parse(await file.text()));
      else if (/\.(glb|gltf)$/i.test(file.name)) select((await addObject(await file.arrayBuffer(), file.name)).info.id);
      else say(`${file.name}: only .glb and .gltf objects, or a saved layout (.json)`);
    } catch (error) {
      say(`Could not add ${file.name}: ${error.message}`);
      console.error(error);
    }
  }
}

// ------------------------------------------------------------------ colours
// A piece's colour keeps its light and shade: each splat and each plain part
// of its model takes the chosen colour at its own brightness relative to the
// piece's average, so a bed's shadowed side stays darker than its top. A
// photographed side (a texture) can only be multiplied, so it is tinted
// towards the colour instead. Its own colours are kept to go back to.
const luma = (c) => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;

function materialsOf(root) {
  const found = new Set();
  root.traverse((node) => {
    if (!node.isMesh) return;
    for (const material of [node.material].flat()) if (material?.color) found.add(material);
  });
  return [...found];
}

const averages = new WeakMap();   // texture -> its average colour, linear
function textureAverage(map) {
  if (averages.has(map)) return averages.get(map);
  let colour = new THREE.Color(1, 1, 1);
  try {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 8;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(map.image, 0, 0, 8, 8);
    const px = context.getImageData(0, 0, 8, 8).data;
    let r = 0, g = 0, b = 0;
    for (let i = 0; i < px.length; i += 4) { r += px[i]; g += px[i + 1]; b += px[i + 2]; }
    colour = new THREE.Color().setRGB(r / 64 / 255, g / 64 / 255, b / 64 / 255, THREE.SRGBColorSpace);
  } catch { /* an image that cannot be drawn counts as white */ }
  averages.set(map, colour);
  return colour;
}

// The piece's own colours, read once: each model's materials (linear, as three.js
// multiplies them) and the splat's colours (as stored), with their averages.
async function ownColours(piece) {
  if (piece.own) return piece.own;
  await piece.modelReady;
  const models = [piece.model, piece.generated].filter(Boolean).map((root) => {
    const materials = materialsOf(root);
    let sum = 0;
    for (const material of materials) {
      material.userData.originalColor ??= material.color.clone();
      const colour = material.userData.originalColor.clone();
      if (material.map) colour.multiply(textureAverage(material.map));
      sum += luma(colour);
    }
    return { materials, luma: materials.length ? sum / materials.length : 0 };
  });
  const own = { models, splat: null, splatLuma: 0 };
  const packed = piece.splat && (await piece.splat.initialized, piece.splat.packedSplats);
  if (packed?.numSplats) {
    const rgb = new Float32Array(packed.numSplats * 3);
    let lumaSum = 0, weight = 0;
    packed.forEachSplat((i, center, scales, quaternion, opacity, color) => {
      rgb[3 * i] = color.r; rgb[3 * i + 1] = color.g; rgb[3 * i + 2] = color.b;
      lumaSum += luma(color) * opacity; weight += opacity;
    });
    own.splat = rgb;
    own.splatLuma = weight ? lumaSum / weight : 0;
  }
  piece.own = own;
  return own;
}

async function setTint(id, hex) {
  const piece = pieces.get(id);
  if (!piece) return;
  piece.tint = hex || null;
  if (!hex && !piece.own) return;                  // never coloured: it still has its own colours
  const own = await ownColours(piece);
  if (piece.tint !== (hex || null)) return;          // changed again while the splat loaded
  if (own.splat) {
    const packed = piece.splat.packedSplats;
    const target = hex ? new THREE.Color().setRGB(...srgb(hex)) : null;
    const k = target ? 1 / Math.max(own.splatLuma, 0.02) : 0;
    const colour = new THREE.Color();
    packed.forEachSplat((i, center, scales, quaternion, opacity) => {
      const r = own.splat[3 * i], g = own.splat[3 * i + 1], b = own.splat[3 * i + 2];
      if (target) {
        const level = (0.2126 * r + 0.7152 * g + 0.0722 * b) * k;
        colour.setRGB(Math.min(1, target.r * level), Math.min(1, target.g * level), Math.min(1, target.b * level));
      } else {
        colour.setRGB(r, g, b);
      }
      packed.setSplat(i, center, scales, quaternion, opacity, colour);
    });
    packed.needsUpdate = true;
    piece.splat.updateVersion();
  }
  const target = hex ? new THREE.Color(hex) : null;
  for (const { materials, luma: average } of own.models) {
    for (const material of materials) {
      const original = material.userData.originalColor;
      material.color.copy(original);
      if (!target) continue;
      if (material.map) {
        // Multiplied towards the colour, as far as the photograph's own colours allow.
        const mean = textureAverage(material.map);
        const k = (t, m) => THREE.MathUtils.clamp(t / Math.max(m, 0.02), 0, 4);
        material.color.setRGB(k(target.r, mean.r), k(target.g, mean.g), k(target.b, mean.b));
      } else {
        material.color.copy(target).multiplyScalar(luma(original) / Math.max(average, 0.02));
      }
    }
  }
  piece.row.querySelector(".swatch").style.background = hex || "";
  if (selected === id) showTint(piece);
}

// A hex colour's sRGB components, 0..1: the space a splat's colours are stored in.
function srgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
}

function showTint(piece) {
  document.getElementById("tint").value = piece.tint || "#c8b49a";
  document.getElementById("untint").hidden = !piece.tint;
}

// ---------------------------------------------------------------- selection
function select(id) {
  selected = id;
  for (const piece of pieces.values()) piece.row.classList.toggle("selected", piece.info.id === id);
  const bar = document.getElementById("selection");
  if (!id) { outline.visible = false; bar.hidden = true; return; }
  const piece = pieces.get(id);
  document.getElementById("sel-title").textContent = piece.info.label;
  const carried = [...pieces.values()].filter((p) => p.info.on === id).map((p) => p.info.label);
  const on = piece.info.on && pieces.get(piece.info.on);
  const size = piece.size.clone().multiplyScalar(piece.group.scale.x);
  document.getElementById("sel-size").textContent =
    `${size.x.toFixed(2)} × ${size.z.toFixed(2)} × ${size.y.toFixed(2)} m`
    + (on ? ` · on the ${on.info.label.toLowerCase()}` : "")
    + (carried.length ? ` · carries the ${carried.join(", ").toLowerCase()}` : "");
  document.getElementById("sel-size").title =
    piece.info.onWhy || (carried.length ? "these move and hide with it" : "");
  const swap = document.getElementById("swap");
  swap.hidden = Object.keys(piece.looks).length < 2;
  swap.textContent = SHOW[nextLook(piece)];
  swap.title = [piece.info.why, piece.info.generated &&
    `generated model: agrees ${piece.info.generated.fit.iou.toFixed(2)} with the video's frames, the plain model ${piece.info.generated.fit.boxIou.toFixed(2)}`]
    .filter(Boolean).join(" · ");
  showTint(piece);
  document.getElementById("height-wrap").hidden = !piece.info.imported;
  document.getElementById("height").value = Math.round(size.y * 100);
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

const SHOW = { scan: "Show the scan", model: "Show clean model", generated: "Show generated model" };
function nextLook(piece) {
  const kinds = Object.keys(piece.looks);
  return kinds[(kinds.indexOf(piece.showing) + 1) % kinds.length];
}
function showAs(id, what) {
  const piece = pieces.get(id);
  if (!piece || !piece.looks[what] || piece.info.imported) return;
  piece.showing = what;
  for (const [kind, look] of Object.entries(piece.looks)) look.visible = kind === what;
  piece.row.querySelector(".count").textContent = what;
  if (selected === id) document.getElementById("swap").textContent = SHOW[nextLook(piece)];
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
document.getElementById("swap").onclick = () => selected && showAs(selected, nextLook(pieces.get(selected)));
document.getElementById("all-models").onclick = () => {
  for (const p of pieces.values()) showAs(p.info.id, p.looks.generated ? "generated" : "model");
};
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
// The layout: where every movable piece is, how it shows, its colour; the
// added objects go in whole (their GLB as a data: URL), so the file alone
// brings the room back.
function currentLayout() {
  const movable = [...pieces.values()].filter((p) => p.info.movable);
  return {
    space: manifest.space,
    added: movable.filter((p) => p.info.imported).map((p) => ({ id: p.info.id, label: p.info.label, glb: p.glb })),
    pieces: movable.map((p) => {
      const position = p.group.getWorldPosition(new THREE.Vector3());
      const entry = { id: p.info.id, label: p.info.label, hidden: !p.group.visible, show: p.showing,
                      position: position.toArray().map((v) => +v.toFixed(3)),
                      turnDegrees: +THREE.MathUtils.radToDeg(p.group.rotation.y).toFixed(1) };
      if (p.tint) entry.color = p.tint;
      if (p.group.scale.x !== 1) entry.scale = +p.group.scale.x.toFixed(4);
      return entry;
    }),
  };
}

async function applyLayout(layout) {
  if (layout.space && layout.space !== manifest.space) say(`This layout was saved for ${layout.space}; applying it by piece`);
  for (const item of layout.added ?? []) {
    if (pieces.has(item.id)) continue;
    const data = await (await fetch(item.glb)).arrayBuffer();
    const piece = await addObject(data, item.label);
    if (piece.info.id !== item.id) {                // keep the saved id, so its entry below finds it
      pieces.delete(piece.info.id);
      piece.info.id = item.id;
      piece.proxy.userData.piece = item.id;
      pieces.set(item.id, piece);
    }
  }
  // Pieces standing on the floor first: what stands on them is placed in their new frame.
  const entries = [...(layout.pieces ?? [])].sort((a, b) => !!pieces.get(a.id)?.info.on - !!pieces.get(b.id)?.info.on);
  let missing = 0;
  for (const entry of entries) {
    const piece = pieces.get(entry.id);
    if (!piece) { missing += 1; continue; }
    if (entry.scale) piece.group.scale.setScalar(entry.scale);
    piece.group.rotation.y = THREE.MathUtils.degToRad(entry.turnDegrees ?? 0);
    if (entry.position) {
      piece.group.parent.updateWorldMatrix(true, false);
      piece.group.position.copy(piece.group.parent.worldToLocal(new THREE.Vector3(...entry.position)));
    }
    piece.group.updateMatrixWorld(true);
    show(entry.id, !entry.hidden);
    if (entry.show) showAs(entry.id, entry.show);
    await setTint(entry.id, entry.color ?? null);
  }
  if (selected) select(selected);
  say(`Layout opened${missing ? `; ${missing} piece(s) it names are not in this scene` : ""}`);
}

document.getElementById("export").onclick = () => {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([JSON.stringify(currentLayout(), null, 1)], { type: "application/json" }));
  link.download = `${manifest.space}-layout.json`;
  link.click();
};
const picker = document.getElementById("picker");
document.getElementById("add").onclick = () => { picker.accept = ".glb,.gltf"; picker.click(); };
document.getElementById("load").onclick = () => { picker.accept = ".json"; picker.click(); };
picker.onchange = () => { addFiles([...picker.files]); picker.value = ""; };
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => { event.preventDefault(); addFiles([...event.dataTransfer.files]); });
document.getElementById("tint").oninput = (event) => selected && setTint(selected, event.target.value);
document.getElementById("untint").onclick = () => selected && setTint(selected, null);
document.getElementById("height").onchange = (event) => {
  const piece = selected && pieces.get(selected);
  const metres = Number(event.target.value) / 100;
  if (!piece?.info.imported || !(metres > 0)) return;
  piece.group.scale.setScalar(metres / piece.size.y);
  select(selected);
};
window.addEventListener("keydown", (event) => {
  if (event.target.closest?.("input")) return;     // typing a height or a colour
  if (event.key === "q" || event.key === "Q") turn(15);
  if (event.key === "e" || event.key === "E") turn(-15);
  if (event.key === "h" || event.key === "H" || event.key === "Delete" || event.key === "Backspace") hideSelected();
  if (event.key === "m" || event.key === "M") selected && showAs(selected, nextLook(pieces.get(selected)));
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

// For checks from outside (the apps, tools/tests/test_scene_viewer.py).
window.oasisScene = { pieces, scene, camera, select, addObject, setTint, currentLayout, applyLayout,
                      ready: () => status.classList.contains("done") };
load().catch((error) => { status.textContent = `Could not load the scene: ${error.message}`; console.error(error); });
