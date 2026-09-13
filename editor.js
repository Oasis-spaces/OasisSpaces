// The interactive previews: a room you can rearrange, and the same room
// restyled by a suggestion, with a before/after switch.
import * as THREE from 'three';
import { createScene, createTweens, easeOut, easeInOut, clamp } from './scene.js';
import { THEMES, LAYOUTS, ITEMS, ROOM, SUGGESTIONS, buildItem, clampToRoom, footprint, setTheme, tickMaterials } from './room.js';

const deg = THREE.MathUtils.degToRad;

// A set of furniture driven by plain data: {id, type, x, z, rot, s}.
function createModel(S, tweens) {
  const items = new Map(); // id -> { data, group }

  function place(entry, animate) {
    const { data, group } = entry;
    const to = { x: data.x, z: data.z, rot: deg(data.rot || 0), s: data.s || 1 };
    if (!animate) {
      group.position.set(to.x, 0, to.z);
      group.rotation.y = to.rot;
      group.scale.setScalar(to.s);
      return;
    }
    const from = { x: group.position.x, z: group.position.z, rot: group.rotation.y, s: group.scale.x };
    // turn the short way round
    let dr = to.rot - from.rot;
    dr = Math.atan2(Math.sin(dr), Math.cos(dr));
    tweens.add('place:' + data.id, 0.7, (e) => {
      const k = easeInOut(e);
      group.position.set(from.x + (to.x - from.x) * k, 0, from.z + (to.z - from.z) * k);
      group.rotation.y = from.rot + dr * k;
      group.scale.setScalar(from.s + (to.s - from.s) * k);
    }, (t) => t);
  }

  function add(data, animate = true) {
    const group = buildItem(data.type, S.mats);
    group.userData.id = data.id;
    S.scene.add(group);
    const entry = { data: { rot: 0, s: 1, ...data }, group };
    items.set(data.id, entry);
    place(entry, false);
    if (animate) {
      const s = entry.data.s;
      tweens.add('grow:' + data.id, 0.5, (e) => group.scale.setScalar(Math.max(0.001, s * e)));
    }
    return entry;
  }

  function remove(id, animate = true) {
    const entry = items.get(id);
    if (!entry) return;
    items.delete(id);
    const g = entry.group;
    const s = g.scale.x;
    const gone = () => { S.scene.remove(g); g.traverse((o) => o.geometry?.dispose()); };
    if (!animate) return gone();
    tweens.add('shrink:' + id, 0.35, (e) => {
      g.scale.setScalar(Math.max(0.001, s * (1 - e)));
      if (e >= 1) gone();
    });
  }

  // Make the scene match a layout list, moving what exists and adding or
  // removing the rest.
  function apply(list, animate = true) {
    const keep = new Set(list.map((d) => d.id));
    for (const id of [...items.keys()]) if (!keep.has(id)) remove(id, animate);
    for (const d of list) {
      const entry = items.get(d.id);
      if (entry && entry.data.type === d.type) {
        entry.data = { rot: 0, s: 1, ...d };
        place(entry, animate);
      } else {
        if (entry) remove(d.id, animate);
        add(d, animate);
      }
    }
  }

  const snapshot = () => [...items.values()].map((e) => ({ ...e.data }));
  return { items, add, remove, apply, place, snapshot };
}

// A rounded outline drawn on the floor under the selected piece.
function frameGeometry(w, d, r = 0.1, t = 0.02) {
  const rr = (hw, hd, rad) => {
    const s = new THREE.Shape();
    s.moveTo(-hw + rad, -hd);
    s.lineTo(hw - rad, -hd); s.quadraticCurveTo(hw, -hd, hw, -hd + rad);
    s.lineTo(hw, hd - rad); s.quadraticCurveTo(hw, hd, hw - rad, hd);
    s.lineTo(-hw + rad, hd); s.quadraticCurveTo(-hw, hd, -hw, hd - rad);
    s.lineTo(-hw, -hd + rad); s.quadraticCurveTo(-hw, -hd, -hw + rad, -hd);
    return s;
  };
  const outer = rr(w / 2, d / 2, r);
  outer.holes.push(rr(w / 2 - t, d / 2 - t, Math.max(0.01, r - t)));
  const g = new THREE.ShapeGeometry(outer, 6);
  g.rotateX(-Math.PI / 2);
  return g;
}

export function createEditor(canvas, ui) {
  const S = createScene(canvas, THEMES.original);
  const tweens = createTweens();
  const model = createModel(S, tweens);
  model.apply(LAYOUTS.original, false);

  const frame = new THREE.Mesh(frameGeometry(1, 1), new THREE.MeshBasicMaterial({
    color: 0x1e5a4c, transparent: true, opacity: 0.9, depthWrite: false,
  }));
  frame.position.y = 0.022; // above a rug, below any furniture base
  frame.visible = false;
  S.scene.add(frame);

  const undo = [];
  let selected = null; // entry
  let drag = null;
  let hoverId = null;
  const hit = new THREE.Vector3();

  function setHint(text) { if (ui.hint) ui.hint.textContent = text; }
  function pushUndo() { undo.push(model.snapshot()); if (undo.length > 40) undo.shift(); ui.undo.disabled = false; }

  function refreshFrame() {
    if (!selected) { frame.visible = false; return; }
    const { data, group } = selected;
    const spec = ITEMS[data.type];
    frame.geometry.dispose();
    frame.geometry = frameGeometry(spec.w * group.scale.x + 0.18, spec.d * group.scale.x + 0.18);
    frame.position.set(group.position.x, 0.022, group.position.z);
    frame.rotation.y = group.rotation.y;
    frame.visible = true;
  }

  function select(entry) {
    selected = entry;
    for (const b of ui.needSelection) b.disabled = !entry;
    if (entry) {
      const spec = ITEMS[entry.data.type];
      const s = entry.data.s || 1; // not the group's scale: a new piece is still growing in
      ui.label.textContent = `${spec.label} · ${(spec.w * s).toFixed(1)} × ${(spec.d * s).toFixed(1)} m`;
      ui.label.hidden = false;
      setHint('Drag to move. Rotate, resize or remove it from the toolbar.');
    } else {
      ui.label.hidden = true;
      setHint('Click a piece to select it, then drag it across the floor.');
    }
    refreshFrame();
  }

  function itemAt(e) {
    const groups = [...model.items.values()].map((v) => v.group);
    const hits = S.cast(e).intersectObjects(groups, true);
    if (!hits.length) return null;
    let o = hits[0].object;
    while (o && o.userData.id === undefined) o = o.parent;
    return o ? model.items.get(o.userData.id) : null;
  }

  canvas.addEventListener('pointerdown', (e) => {
    const entry = itemAt(e);
    canvas.setPointerCapture(e.pointerId);
    if (entry) {
      select(entry);
      S.floorHit(e, hit);
      drag = { entry, dx: entry.group.position.x - hit.x, dz: entry.group.position.z - hit.z, moved: false, x: e.clientX, y: e.clientY };
    } else {
      select(null);
      drag = { orbit: true, x: e.clientX, y: e.clientY };
    }
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) {
      const entry = itemAt(e);
      const id = entry ? entry.data.id : null;
      if (id !== hoverId) { hoverId = id; canvas.style.cursor = id ? 'grab' : ''; }
      return;
    }
    if (drag.orbit) {
      S.orbitBy(e.clientX - drag.x, e.clientY - drag.y);
      drag.x = e.clientX; drag.y = e.clientY;
      return;
    }
    if (!S.floorHit(e, hit)) return;
    if (!drag.moved) {
      if (Math.hypot(e.clientX - drag.x, e.clientY - drag.y) < 3) return;
      drag.moved = true;
      pushUndo();
      canvas.style.cursor = 'grabbing';
    }
    const { entry } = drag;
    const spec = ITEMS[entry.data.type];
    const [x, z] = clampToRoom(hit.x + drag.dx, hit.z + drag.dz, spec, entry.data.rot, entry.group.scale.x);
    entry.group.position.set(x, 0, z);
    refreshFrame();
  });
  const endDrag = () => {
    if (drag && drag.entry && drag.moved) {
      drag.entry.data.x = drag.entry.group.position.x;
      drag.entry.data.z = drag.entry.group.position.z;
    }
    drag = null;
    canvas.style.cursor = hoverId ? 'grab' : '';
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);

  // ------------------------------------------------------------- toolbar
  function rotate() {
    if (!selected) return;
    pushUndo();
    const { data, group } = selected;
    const spec = ITEMS[data.type];
    data.rot = ((data.rot || 0) + 90) % 360;
    [data.x, data.z] = clampToRoom(group.position.x, group.position.z, spec, data.rot, group.scale.x);
    model.place(selected, true);
  }
  function resize(factor) {
    if (!selected) return;
    const { data, group } = selected;
    const s = clamp((data.s || 1) * factor, 0.7, 1.4);
    if (s === data.s) return;
    pushUndo();
    data.s = s;
    const spec = ITEMS[data.type];
    [data.x, data.z] = clampToRoom(group.position.x, group.position.z, spec, data.rot, s);
    model.place(selected, true);
    ui.label.textContent = `${spec.label} · ${(spec.w * s).toFixed(1)} × ${(spec.d * s).toFixed(1)} m`;
  }
  function removeSelected() {
    if (!selected) return;
    pushUndo();
    model.remove(selected.data.id, true);
    select(null);
  }
  let counter = 0;
  function addItem(type) {
    pushUndo();
    const spec = ITEMS[type];
    // drop it on free floor near the open side of the room
    const [x, z] = clampToRoom(0.6, 1.1, spec, 0, 1);
    const entry = model.add({ id: `${type}-${++counter}`, type, x, z, rot: 0, s: 1 }, true);
    select(entry);
    ui.addMenu.hidden = true;
  }
  function undoLast() {
    const snap = undo.pop();
    if (!snap) return;
    select(null);
    model.apply(snap, true);
    ui.undo.disabled = undo.length === 0;
  }

  ui.add.addEventListener('click', () => { ui.addMenu.hidden = !ui.addMenu.hidden; });
  ui.addMenu.addEventListener('click', (e) => {
    const b = e.target.closest('[data-type]');
    if (b) addItem(b.dataset.type);
  });
  ui.rotate.addEventListener('click', rotate);
  ui.smaller.addEventListener('click', () => resize(1 / 1.1));
  ui.larger.addEventListener('click', () => resize(1.1));
  ui.remove.addEventListener('click', removeSelected);
  ui.undo.addEventListener('click', undoLast);
  ui.undo.disabled = true;
  canvas.tabIndex = 0;
  canvas.addEventListener('keydown', (e) => {
    if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); removeSelected(); }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undoLast(); }
    if (e.key.toLowerCase() === 'r') rotate();
    if (e.key === 'Escape') { select(null); ui.addMenu.hidden = true; }
  });
  document.addEventListener('pointerdown', (e) => {
    if (!ui.addMenu.hidden && !ui.addMenu.contains(e.target) && e.target !== ui.add) ui.addMenu.hidden = true;
  });
  select(null);

  const v = new THREE.Vector3();
  function update(dt) {
    tweens.update(dt);
    tickMaterials(S.mats);
    if (selected) {
      if (tweens.busy) refreshFrame();
      const { group, data } = selected;
      v.set(group.position.x, ITEMS[data.type].h * group.scale.x + 0.12, group.position.z).project(S.camera);
      const r = canvas.getBoundingClientRect();
      ui.label.style.left = `${(v.x + 1) / 2 * r.width}px`;
      ui.label.style.top = `${(1 - v.y) / 2 * r.height}px`;
    }
    S.render();
  }
  return { update };
}

export function createRedesign(canvas, ui) {
  const S = createScene(canvas, THEMES.original, { azBase: 0.7, elBase: 0.42 });
  const tweens = createTweens();
  const model = createModel(S, tweens);
  model.apply(LAYOUTS.original, false);

  let style = 'warm';
  let after = true;

  function show() {
    const key = after ? style : 'original';
    setTheme(S.mats, THEMES[key]);
    model.apply(LAYOUTS[key], true);
    for (const b of ui.compare) b.classList.toggle('is-on', (b.dataset.view === 'after') === after);
    for (const c of ui.chips) c.classList.toggle('is-on', c.dataset.style === style);
    c: {
      const s = SUGGESTIONS[style];
      ui.title.textContent = THEMES[style].name;
      ui.text.textContent = s.text;
      ui.list.replaceChildren(...s.changes.map((t) => { const li = document.createElement('li'); li.textContent = t; return li; }));
    }
  }
  for (const c of ui.chips) c.addEventListener('click', () => { style = c.dataset.style; after = true; show(); });
  for (const b of ui.compare) b.addEventListener('click', () => { after = b.dataset.view === 'after'; show(); });

  let drag = null;
  canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    S.orbitBy(e.clientX - drag.x, e.clientY - drag.y);
    drag = { x: e.clientX, y: e.clientY };
  });
  const release = () => { drag = null; };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);

  show();
  function update(dt) {
    tweens.update(dt);
    tickMaterials(S.mats, 0.06);
    S.render();
  }
  return { update, show };
}
