// A renderer, camera and lit room that the hero, editor and redesign scenes
// share. The camera orbits a point in the room within limits that keep the
// open sides of the dollhouse toward the viewer.
import * as THREE from 'three';
import { makeMaterials, buildRoom, ROOM } from './room.js';

const DPR = Math.min(window.devicePixelRatio || 1, 2);
export const clamp = THREE.MathUtils.clamp;
export const easeOut = (t) => 1 - Math.pow(1 - clamp(t, 0, 1), 3);
export const easeInOut = (t) => { t = clamp(t, 0, 1); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };

export function createScene(canvas, theme, view = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(DPR);
  renderer.setClearColor(0x000000, 0);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 80);
  const mats = makeMaterials(theme);
  const room = buildRoom(mats);
  scene.add(room.group);

  scene.add(new THREE.HemisphereLight(0xfff6ea, 0x9a8b78, 0.9));
  const sun = new THREE.DirectionalLight(0xfff0dc, 2.3);
  sun.position.set(3.2, 6.5, 4.2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -4.2, right: 4.2, top: 4.2, bottom: -4.2, near: 1, far: 22 });
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.025;
  sun.shadow.radius = 5;
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0xdfe8ff, 0.45);
  fill.position.set(-4, 3, 5);
  scene.add(fill);

  const orbit = {
    az: 0.62, el: 0.40, r: 7.5, azBase: 0.62, elBase: 0.40, dAz: 0, dEl: 0, pAz: 0, pEl: 0,
    target: new THREE.Vector3(0.05, 0.72, 0.15), ...view,
  };
  const LIM = { az: [0.12, 1.3], el: [0.16, 0.8] };

  function placeCamera() {
    const azT = clamp(orbit.azBase + orbit.dAz + orbit.pAz, LIM.az[0], LIM.az[1]);
    const elT = clamp(orbit.elBase + orbit.dEl + orbit.pEl, LIM.el[0], LIM.el[1]);
    orbit.az += (azT - orbit.az) * 0.08;
    orbit.el += (elT - orbit.el) * 0.08;
    const t = orbit.target;
    camera.position.set(
      t.x + orbit.r * Math.cos(orbit.el) * Math.sin(orbit.az),
      t.y + orbit.r * Math.sin(orbit.el),
      t.z + orbit.r * Math.cos(orbit.el) * Math.cos(orbit.az),
    );
    camera.lookAt(t);
  }

  // The parts of the dollhouse that are actually drawn: the floor, the two
  // far walls, and whatever stands near the open sides.
  const { w: RW, d: RD, h: RH } = ROOM;
  const extent = [
    [-RW / 2 - 0.15, -0.1, -RD / 2 - 0.15], [RW / 2 + 0.15, -0.1, -RD / 2 - 0.15],
    [-RW / 2 - 0.15, -0.1, RD / 2 + 0.15], [RW / 2 + 0.15, -0.1, RD / 2 + 0.15],
    [-RW / 2 - 0.15, RH + 0.1, -RD / 2 - 0.15], [RW / 2 + 0.15, RH + 0.1, -RD / 2 - 0.15],
    [-RW / 2 - 0.15, RH + 0.1, RD / 2 + 0.15], [RW / 2 + 0.15, 1.1, RD / 2 + 0.15],
  ].map((p) => new THREE.Vector3(...p));
  const probe = new THREE.Vector3();
  function positionFor(az, el, r) {
    const t = orbit.target;
    camera.position.set(
      t.x + r * Math.cos(el) * Math.sin(az), t.y + r * Math.sin(el), t.z + r * Math.cos(el) * Math.cos(az));
    camera.lookAt(t);
    camera.updateMatrixWorld();
  }
  // Pull the camera back just far enough that the room fits the stage.
  function fitRoom() {
    let r = 6;
    for (let i = 0; i < 5; i++) {
      positionFor(orbit.azBase, orbit.elBase, r);
      let m = 0;
      for (const c of extent) { probe.copy(c).project(camera); m = Math.max(m, Math.abs(probe.x), Math.abs(probe.y)); }
      r *= m / 0.93;
    }
    orbit.r = r;
  }

  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width)), h = Math.max(1, Math.round(rect.height));
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.fov = w / h < 1 ? 36 : 30;
    camera.updateProjectionMatrix();
    fitRoom();
  }
  new ResizeObserver(resize).observe(canvas.parentElement);
  resize();

  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  function cast(e) {
    const r = canvas.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    return raycaster;
  }
  function floorHit(e, out) { return cast(e).ray.intersectPlane(floorPlane, out); }

  // drag on empty space to look around
  function orbitBy(dx, dy) {
    orbit.dAz = clamp(orbit.dAz - dx * 0.005, LIM.az[0] - orbit.azBase, LIM.az[1] - orbit.azBase);
    orbit.dEl = clamp(orbit.dEl + dy * 0.004, LIM.el[0] - orbit.elBase, LIM.el[1] - orbit.elBase);
  }

  function render() { placeCamera(); renderer.render(scene, camera); }
  return { renderer, scene, camera, mats, room, orbit, cast, floorHit, orbitBy, render, resize, raycaster };
}

// Minimal tweens: each drives a 0..1 value through `apply`.
export function createTweens() {
  const list = [];
  return {
    add(key, dur, apply, ease = easeOut) {
      const i = list.findIndex((t) => t.key === key);
      if (i >= 0) list.splice(i, 1);
      const tw = { key, t: 0, dur, apply, ease, done: false };
      list.push(tw);
      apply(0);
      return tw;
    },
    update(dt) {
      for (const tw of list) {
        tw.t = Math.min(tw.dur, tw.t + dt);
        tw.apply(tw.ease(tw.t / tw.dur));
        if (tw.t >= tw.dur) tw.done = true;
      }
      for (let i = list.length - 1; i >= 0; i--) if (list[i].done) list.splice(i, 1);
    },
    get busy() { return list.length > 0; },
  };
}
