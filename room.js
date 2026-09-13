// The room, its furniture and the colour themes shared by every 3D scene on
// the site. Everything is built from primitives in metres, so a bed here is
// the same 2.0 x 1.6 m a bed is in the real editor.
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

export const ROOM = { w: 4.4, d: 3.8, h: 2.7 };

export const THEMES = {
  original: {
    name: 'Original', wall: '#EBE4D8', skirting: '#F6F2EA', floor: '#B48E63', rug: '#CFC3B0',
    wood: '#6E4F35', wood2: '#8A6A4B', mattress: '#F3EFE7', fabric: '#8E9DA6', pillow: '#F7F4EE',
    shade: '#F1E6CF', metal: '#2A2926', pot: '#A9754E', leaf: '#4D7A4C', frame: '#F6F2EA',
    glass: '#E6EEF0', accent: '#3F5C52',
  },
  warm: {
    name: 'Warm minimal', wall: '#F0E9DD', skirting: '#F8F4EC', floor: '#C7A77C', rug: '#E2D6C2',
    wood: '#9A7754', wood2: '#B39473', mattress: '#F6F2EA', fabric: '#C9B7A0', pillow: '#FBF8F2',
    shade: '#F5EBD6', metal: '#3A3530', pot: '#C99A6F', leaf: '#5E8A55', frame: '#F8F4EC',
    glass: '#EDF2F1', accent: '#8C5A3C',
  },
  coastal: {
    name: 'Coastal', wall: '#F3F1EA', skirting: '#FBFAF6', floor: '#D8C6A7', rug: '#E9E4D6',
    wood: '#B7A388', wood2: '#C8B99F', mattress: '#F8F7F2', fabric: '#8FA9B5', pillow: '#FBFBF8',
    shade: '#F4EEE0', metal: '#5A5F62', pot: '#D9D2C4', leaf: '#6C9A6E', frame: '#FBFAF6',
    glass: '#E3EEF3', accent: '#4A7E8C',
  },
  japandi: {
    name: 'Japandi', wall: '#E7E0D2', skirting: '#EFE9DD', floor: '#A88763', rug: '#D9CDB9',
    wood: '#4B3A2C', wood2: '#5F4A38', mattress: '#F0EBE1', fabric: '#B5AA97', pillow: '#F3EEE4',
    shade: '#EADFC8', metal: '#26221E', pot: '#3E3731', leaf: '#587A52', frame: '#EFE9DD',
    glass: '#E8EDEA', accent: '#5B5A45',
  },
  midcentury: {
    name: 'Mid-century', wall: '#EFE7D6', skirting: '#F7F1E4', floor: '#8F6A45', rug: '#C8B08F',
    wood: '#5C3A22', wood2: '#7A4E2E', mattress: '#F2ECE0', fabric: '#A88A4E', pillow: '#F5F0E4',
    shade: '#F1E1BE', metal: '#2B2521', pot: '#C2643E', leaf: '#4E7847', frame: '#F7F1E4',
    glass: '#E9EEEA', accent: '#B45A3A',
  },
};

const ROUGHNESS = {
  wall: 0.95, skirting: 0.8, floor: 0.72, rug: 1.0, wood: 0.55, wood2: 0.6, mattress: 0.9,
  fabric: 0.95, pillow: 0.9, shade: 0.8, metal: 0.4, pot: 0.8, leaf: 0.85, frame: 0.7, accent: 0.7,
};

export function makeMaterials(theme) {
  const mats = {};
  for (const role of Object.keys(ROUGHNESS)) {
    const m = new THREE.MeshStandardMaterial({
      color: theme[role], roughness: ROUGHNESS[role], metalness: role === 'metal' ? 0.45 : 0,
    });
    if (role === 'leaf') m.flatShading = true;
    m.userData = { role, target: new THREE.Color(theme[role]) };
    mats[role] = m;
  }
  // The window pane is unlit so it reads as daylight.
  mats.glass = new THREE.MeshBasicMaterial({ color: theme.glass });
  mats.glass.userData = { role: 'glass', target: new THREE.Color(theme.glass) };
  return mats;
}

export function setTheme(mats, theme) {
  for (const m of Object.values(mats)) m.userData.target.set(theme[m.userData.role]);
}

// Ease every material toward its theme colour; call once per frame.
export function tickMaterials(mats, k = 0.08) {
  for (const m of Object.values(mats)) {
    const t = m.userData.target;
    if (m.color.equals(t)) continue;
    m.color.lerp(t, k);
    if (Math.abs(m.color.r - t.r) + Math.abs(m.color.g - t.g) + Math.abs(m.color.b - t.b) < 0.004) m.color.copy(t);
  }
}

// ---------------------------------------------------------------- primitives
function rbox(mat, w, h, d, x = 0, y = 0, z = 0, r = 0.02, part = 'furniture') {
  const radius = Math.min(r, w / 2, h / 2, d / 2);
  const mesh = new THREE.Mesh(new RoundedBoxGeometry(w, h, d, 3, radius), mat);
  mesh.position.set(x, y + h / 2, z); // y is the bottom of the box
  mesh.castShadow = mesh.receiveShadow = true;
  mesh.userData.part = part;
  return mesh;
}

function cyl(mat, rTop, rBottom, h, x = 0, y = 0, z = 0, seg = 28) {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(rTop, rBottom, h, seg), mat);
  mesh.position.set(x, y + h / 2, z);
  mesh.castShadow = mesh.receiveShadow = true;
  mesh.userData.part = 'furniture';
  return mesh;
}

function blob(mat, r, x, y, z) {
  const mesh = new THREE.Mesh(new THREE.IcosahedronGeometry(r, 1), mat);
  mesh.position.set(x, y, z);
  mesh.castShadow = mesh.receiveShadow = true;
  mesh.userData.part = 'furniture';
  return mesh;
}

// ----------------------------------------------------------------- furniture
// Each builder returns a group whose origin is the centre of its footprint on
// the floor; w runs along x, d along z, and the piece faces +z.
export const ITEMS = {
  bed: {
    label: 'Bed', w: 1.6, d: 2.0, h: 1.0,
    build(m) {
      const g = new THREE.Group();
      g.add(rbox(m.wood, 1.7, 1.0, 0.08, 0, 0, -0.96, 0.02));       // headboard
      g.add(rbox(m.wood2, 1.6, 0.26, 1.9, 0, 0, 0.02, 0.03));       // base
      g.add(rbox(m.mattress, 1.5, 0.22, 1.84, 0, 0.26, 0.02, 0.06)); // mattress
      g.add(rbox(m.fabric, 1.54, 0.09, 1.25, 0, 0.47, 0.32, 0.05));  // duvet
      g.add(rbox(m.pillow, 0.62, 0.13, 0.42, -0.4, 0.48, -0.62, 0.06));
      g.add(rbox(m.pillow, 0.62, 0.13, 0.42, 0.4, 0.48, -0.62, 0.06));
      return g;
    },
  },
  wardrobe: {
    label: 'Wardrobe', w: 1.2, d: 0.6, h: 2.1,
    build(m) {
      const g = new THREE.Group();
      g.add(rbox(m.wood2, 1.2, 2.1, 0.6, 0, 0, 0, 0.015));
      g.add(rbox(m.metal, 0.012, 1.9, 0.012, 0, 0.1, 0.3, 0.004));
      g.add(rbox(m.metal, 0.02, 0.18, 0.02, -0.06, 0.95, 0.31, 0.008));
      g.add(rbox(m.metal, 0.02, 0.18, 0.02, 0.06, 0.95, 0.31, 0.008));
      return g;
    },
  },
  sidetable: {
    label: 'Bedside table', w: 0.5, d: 0.45, h: 0.56,
    build(m) {
      const g = new THREE.Group();
      g.add(rbox(m.wood, 0.5, 0.04, 0.45, 0, 0.52, 0, 0.01));
      g.add(rbox(m.wood2, 0.46, 0.16, 0.4, 0, 0.34, 0, 0.01));
      for (const [x, z] of [[-0.21, -0.18], [0.21, -0.18], [-0.21, 0.18], [0.21, 0.18]]) {
        g.add(rbox(m.metal, 0.025, 0.34, 0.025, x, 0, z, 0.006));
      }
      return g;
    },
  },
  lamp: {
    label: 'Floor lamp', w: 0.46, d: 0.46, h: 1.58,
    build(m) {
      const g = new THREE.Group();
      g.add(cyl(m.metal, 0.16, 0.17, 0.02));
      g.add(cyl(m.metal, 0.012, 0.012, 1.28, 0, 0.02, 0, 12));
      g.add(cyl(m.shade, 0.17, 0.23, 0.32, 0, 1.26, 0));
      return g;
    },
  },
  plant: {
    label: 'Plant', w: 0.5, d: 0.5, h: 0.95,
    build(m) {
      const g = new THREE.Group();
      g.add(cyl(m.pot, 0.17, 0.13, 0.32));
      g.add(blob(m.leaf, 0.22, 0.02, 0.6, 0.0));
      g.add(blob(m.leaf, 0.18, -0.14, 0.72, 0.08));
      g.add(blob(m.leaf, 0.17, 0.15, 0.76, -0.07));
      g.add(blob(m.leaf, 0.14, 0.0, 0.84, 0.12));
      return g;
    },
  },
  armchair: {
    label: 'Armchair', w: 0.85, d: 0.85, h: 0.92,
    build(m) {
      const g = new THREE.Group();
      for (const [x, z] of [[-0.36, -0.34], [0.36, -0.34], [-0.36, 0.34], [0.36, 0.34]]) {
        g.add(rbox(m.wood, 0.04, 0.12, 0.04, x, 0, z, 0.01));
      }
      g.add(rbox(m.fabric, 0.85, 0.34, 0.8, 0, 0.1, 0.02, 0.06));
      g.add(rbox(m.fabric, 0.85, 0.5, 0.2, 0, 0.42, -0.3, 0.06));
      g.add(rbox(m.fabric, 0.16, 0.22, 0.8, -0.345, 0.42, 0.02, 0.05));
      g.add(rbox(m.fabric, 0.16, 0.22, 0.8, 0.345, 0.42, 0.02, 0.05));
      return g;
    },
  },
  sofa: {
    label: 'Sofa', w: 2.0, d: 0.9, h: 0.96,
    build(m) {
      const g = new THREE.Group();
      for (const [x, z] of [[-0.9, -0.36], [0.9, -0.36], [-0.9, 0.36], [0.9, 0.36]]) {
        g.add(rbox(m.wood, 0.04, 0.12, 0.04, x, 0, z, 0.01));
      }
      g.add(rbox(m.fabric, 2.0, 0.34, 0.85, 0, 0.1, 0.02, 0.06));
      g.add(rbox(m.pillow, 0.9, 0.12, 0.7, -0.48, 0.44, 0.05, 0.05));
      g.add(rbox(m.pillow, 0.9, 0.12, 0.7, 0.48, 0.44, 0.05, 0.05));
      g.add(rbox(m.fabric, 2.0, 0.52, 0.22, 0, 0.44, -0.31, 0.06));
      g.add(rbox(m.fabric, 0.18, 0.22, 0.85, -0.91, 0.44, 0.02, 0.05));
      g.add(rbox(m.fabric, 0.18, 0.22, 0.85, 0.91, 0.44, 0.02, 0.05));
      return g;
    },
  },
  rug: {
    label: 'Rug', w: 2.6, d: 1.8, h: 0.015,
    build(m) {
      const g = new THREE.Group();
      const r = rbox(m.rug, 2.6, 0.015, 1.8, 0, 0, 0, 0.005);
      r.castShadow = false;
      g.add(r);
      return g;
    },
  },
  desk: {
    label: 'Desk', w: 1.3, d: 0.65, h: 0.755,
    build(m) {
      const g = new THREE.Group();
      g.add(rbox(m.wood, 1.3, 0.035, 0.65, 0, 0.72, 0, 0.01));
      for (const [x, z] of [[-0.6, -0.28], [0.6, -0.28], [-0.6, 0.28], [0.6, 0.28]]) {
        g.add(rbox(m.metal, 0.03, 0.72, 0.03, x, 0, z, 0.008));
      }
      return g;
    },
  },
  chair: {
    label: 'Chair', w: 0.48, d: 0.5, h: 0.91,
    build(m) {
      const g = new THREE.Group();
      g.add(rbox(m.wood2, 0.46, 0.04, 0.46, 0, 0.45, 0, 0.012));
      g.add(rbox(m.wood2, 0.44, 0.42, 0.03, 0, 0.49, -0.215, 0.012));
      for (const [x, z] of [[-0.2, -0.2], [0.2, -0.2], [-0.2, 0.2], [0.2, 0.2]]) {
        g.add(rbox(m.metal, 0.025, 0.45, 0.025, x, 0, z, 0.006));
      }
      return g;
    },
  },
  shelf: {
    label: 'Open shelf', w: 1.0, d: 0.32, h: 1.7,
    build(m) {
      const g = new THREE.Group();
      g.add(rbox(m.wood, 0.03, 1.7, 0.32, -0.485, 0, 0, 0.006));
      g.add(rbox(m.wood, 0.03, 1.7, 0.32, 0.485, 0, 0, 0.006));
      g.add(rbox(m.wood2, 0.98, 1.7, 0.012, 0, 0, -0.154, 0.004));
      for (let k = 0; k < 5; k++) g.add(rbox(m.wood, 1.0, 0.03, 0.32, 0, k * 0.415, 0, 0.006));
      // a few things on the shelves
      g.add(rbox(m.accent, 0.05, 0.28, 0.2, -0.3, 0.445, 0.02, 0.006));
      g.add(rbox(m.pot, 0.06, 0.24, 0.2, -0.23, 0.445, 0.02, 0.006));
      g.add(rbox(m.fabric, 0.22, 0.22, 0.22, 0.25, 0.86, 0.0, 0.03));
      g.add(cyl(m.pot, 0.07, 0.06, 0.12, -0.25, 1.275, 0.02));
      g.add(blob(m.leaf, 0.1, -0.25, 1.46, 0.02));
      return g;
    },
  },
};

export function buildItem(type, mats) {
  const spec = ITEMS[type];
  const g = spec.build(mats);
  g.userData = { type, label: spec.label, w: spec.w, d: spec.d, h: spec.h };
  return g;
}

// ---------------------------------------------------------------------- room
// Three sides of a room seen like a dollhouse: floor, back wall with a window,
// left wall. The two sides nearest the camera are left open.
export function buildRoom(m) {
  const { w, d, h } = ROOM;
  const g = new THREE.Group();
  const t = 0.08; // wall thickness
  const floor = rbox(m.floor, w + t, 0.06, d + t, -t / 2, -0.06, -t / 2, 0.01, 'floor');
  g.add(floor);

  const left = rbox(m.wall, t, h, d + t, -w / 2 - t / 2, 0, -t / 2, 0.01, 'wall');
  g.add(left);
  // back wall around a 1.4 x 1.3 m window
  const [wx0, wx1, wy0, wy1] = [-0.5, 0.9, 0.95, 2.25];
  const zb = -d / 2 - t / 2;
  g.add(rbox(m.wall, wx0 + w / 2, h, t, (-w / 2 + wx0) / 2, 0, zb, 0.01, 'wall'));
  g.add(rbox(m.wall, w / 2 - wx1, h, t, (wx1 + w / 2) / 2, 0, zb, 0.01, 'wall'));
  g.add(rbox(m.wall, wx1 - wx0, wy0, t, (wx0 + wx1) / 2, 0, zb, 0.01, 'wall'));
  g.add(rbox(m.wall, wx1 - wx0, h - wy1, t, (wx0 + wx1) / 2, wy1, zb, 0.01, 'wall'));
  // window: pane, frame and a cross bar
  const pane = new THREE.Mesh(new THREE.PlaneGeometry(wx1 - wx0, wy1 - wy0), m.glass);
  pane.position.set((wx0 + wx1) / 2, (wy0 + wy1) / 2, zb);
  pane.userData.part = 'wall';
  g.add(pane);
  const fw = 0.05;
  g.add(rbox(m.frame, wx1 - wx0 + 2 * fw, fw, t + 0.02, (wx0 + wx1) / 2, wy0 - fw, zb, 0.01, 'wall'));
  g.add(rbox(m.frame, wx1 - wx0 + 2 * fw, fw, t + 0.02, (wx0 + wx1) / 2, wy1, zb, 0.01, 'wall'));
  g.add(rbox(m.frame, fw, wy1 - wy0, t + 0.02, wx0 - fw / 2, wy0, zb, 0.01, 'wall'));
  g.add(rbox(m.frame, fw, wy1 - wy0, t + 0.02, wx1 + fw / 2, wy0, zb, 0.01, 'wall'));
  g.add(rbox(m.frame, 0.03, wy1 - wy0, t + 0.01, (wx0 + wx1) / 2, wy0, zb, 0.006, 'wall'));
  g.add(rbox(m.frame, wx1 - wx0, 0.03, t + 0.01, (wx0 + wx1) / 2, (wy0 + wy1) / 2, zb, 0.006, 'wall'));
  // skirting
  g.add(rbox(m.skirting, 0.015, 0.1, d, -w / 2 + 0.0075, 0, 0, 0.004, 'wall'));
  g.add(rbox(m.skirting, w, 0.1, 0.015, 0, 0, -d / 2 + 0.0075, 0.004, 'wall'));

  // a soft contact shadow so the room sits on the page rather than floating
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const ctx = c.getContext('2d');
  const grad = ctx.createRadialGradient(128, 128, 20, 128, 128, 128);
  grad.addColorStop(0, 'rgba(30,25,20,0.32)');
  grad.addColorStop(0.55, 'rgba(30,25,20,0.14)');
  grad.addColorStop(1, 'rgba(30,25,20,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 256, 256);
  const contact = new THREE.Mesh(
    new THREE.PlaneGeometry(w + 2.4, d + 2.4),
    new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }),
  );
  contact.rotation.x = -Math.PI / 2;
  contact.position.set(0, -0.07, 0.1);
  contact.userData.part = 'contact';
  g.add(contact);
  return { group: g, floor };
}

// Footprint half-extents of an item after its rotation, for keeping it inside.
export function footprint(spec, rotDeg, s = 1) {
  const a = THREE.MathUtils.degToRad(rotDeg);
  const c = Math.abs(Math.cos(a)), n = Math.abs(Math.sin(a));
  return { hx: (c * spec.w + n * spec.d) * s / 2, hz: (n * spec.w + c * spec.d) * s / 2 };
}

export function clampToRoom(x, z, spec, rotDeg, s = 1) {
  const { hx, hz } = footprint(spec, rotDeg, s);
  return [
    THREE.MathUtils.clamp(x, -ROOM.w / 2 + hx, ROOM.w / 2 - hx),
    THREE.MathUtils.clamp(z, -ROOM.d / 2 + hz, ROOM.d / 2 - hz),
  ];
}

// ------------------------------------------------------------------- layouts
export const LAYOUTS = {
  original: [
    { id: 'bed', type: 'bed', x: -0.9, z: -0.88, rot: 0 },
    { id: 'side', type: 'sidetable', x: 0.15, z: -1.65, rot: 0 },
    { id: 'ward', type: 'wardrobe', x: -1.9, z: 0.9, rot: 90 },
    { id: 'rug', type: 'rug', x: 0.3, z: 0.65, rot: 0 },
    { id: 'chair', type: 'armchair', x: 1.5, z: 0.95, rot: -35 },
    { id: 'plant', type: 'plant', x: 1.9, z: -1.5, rot: 0 },
    { id: 'lamp', type: 'lamp', x: 1.95, z: 0.1, rot: 0 },
  ],
  warm: [
    { id: 'bed', type: 'bed', x: 0.2, z: -0.88, rot: 0 },
    { id: 'side', type: 'sidetable', x: 1.25, z: -1.65, rot: 0 },
    { id: 'shelf', type: 'shelf', x: -2.03, z: 0.2, rot: 90 },
    { id: 'rug', type: 'rug', x: 0.2, z: 0.75, rot: 0 },
    { id: 'chair', type: 'armchair', x: 1.6, z: 1.15, rot: -40 },
    { id: 'lamp', type: 'lamp', x: 1.95, z: 0.45, rot: 0 },
    { id: 'plant', type: 'plant', x: -1.9, z: -1.55, rot: 0 },
  ],
  coastal: [
    { id: 'bed', type: 'bed', x: -0.9, z: -0.88, rot: 0 },
    { id: 'side', type: 'sidetable', x: 0.15, z: -1.65, rot: 0 },
    { id: 'ward', type: 'wardrobe', x: -1.9, z: 0.9, rot: 90 },
    { id: 'rug', type: 'rug', x: 0.4, z: 0.7, rot: 0, s: 1.1 },
    { id: 'chair', type: 'armchair', x: 1.45, z: 0.35, rot: -150 },
    { id: 'plant', type: 'plant', x: 1.9, z: -1.5, rot: 0 },
    { id: 'plant2', type: 'plant', x: -1.9, z: 1.6, rot: 0 },
    { id: 'lamp', type: 'lamp', x: 1.95, z: 1.2, rot: 0 },
  ],
  japandi: [
    { id: 'bed', type: 'bed', x: -0.9, z: -0.88, rot: 0 },
    { id: 'side', type: 'sidetable', x: 0.15, z: -1.65, rot: 0 },
    { id: 'side2', type: 'sidetable', x: -1.93, z: -1.65, rot: 0 },
    { id: 'shelf', type: 'shelf', x: 1.6, z: -1.72, rot: 0 },
    { id: 'chair', type: 'armchair', x: 1.4, z: 1.0, rot: -30 },
    { id: 'lamp', type: 'lamp', x: 1.95, z: 0.3, rot: 0 },
    { id: 'plant', type: 'plant', x: -1.9, z: 1.55, rot: 0 },
  ],
  midcentury: [
    { id: 'bed', type: 'bed', x: -0.9, z: -0.88, rot: 0 },
    { id: 'side', type: 'sidetable', x: 0.15, z: -1.65, rot: 0 },
    { id: 'desk', type: 'desk', x: 1.45, z: -1.55, rot: 0 },
    { id: 'deskchair', type: 'chair', x: 1.45, z: -0.95, rot: 180 },
    { id: 'ward', type: 'wardrobe', x: -1.9, z: 0.6, rot: 90 },
    { id: 'rug', type: 'rug', x: 0.2, z: 0.9, rot: 0 },
    { id: 'chair', type: 'armchair', x: 1.55, z: 1.15, rot: -45 },
    { id: 'plant', type: 'plant', x: -1.9, z: 1.62, rot: 0 },
  ],
};

export const SUGGESTIONS = {
  warm: {
    text: 'The bed sits beside the window rather than under it, so the room’s best light lands on empty floor. Centre the bed on the window, trade the wardrobe for a low open shelf to open up the left wall, and move the reading chair into the corner with a floor lamp.',
    changes: ['Bed centred under the window', 'Wardrobe swapped for an open shelf on the left wall', 'Armchair and floor lamp moved into the free corner'],
  },
  coastal: {
    text: 'Lighter surfaces and a cooler palette make the 4.4 m wall read longer than it is. Keep the bed where it stands, turn the armchair toward the window, and bring greenery to both ends of the room.',
    changes: ['Walls and floor lightened, blue-grey textiles', 'Armchair turned to face the window', 'A second plant in the far corner'],
  },
  japandi: {
    text: 'Fewer, lower pieces and darker wood settle the room. Take out the rug and the wardrobe, add a matching bedside table on each side of the bed, and use one tall shelf beside the window for storage.',
    changes: ['Rug and wardrobe removed', 'Bedside tables on both sides of the bed', 'One tall shelf beside the window'],
  },
  midcentury: {
    text: 'The window wall has 1.3 m to spare on the right. Put a desk and chair there for the daylight, keep the wardrobe on the left wall, and tie the pieces together with walnut and olive.',
    changes: ['Desk and chair beside the window', 'Walnut wood, olive textiles', 'Rug moved to the middle of the room'],
  },
};
