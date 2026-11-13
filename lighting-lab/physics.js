// The physics behind the lighting lab, in SI units: lumens, lux (lm/m²), candela,
// nits (cd/m²), watts, volts, amps. Nothing here draws; lab.js turns it into a picture.
//
// - sunPosition: where the sun is for a place and a moment (NOAA's solar equations).
// - daylight: how much light the sun and sky give outside (IES clear/overcast sky formulas).
// - BULBS and bulbState: what a lamp gives and draws at a supply voltage and dimmer setting.
// - Radiosity: the light in the room, direct and bounced, on patches about 20 cm across.
// - ugr: discomfort glare from the lamps in view (CIE 117 Unified Glare Rating).

export const DEG = Math.PI / 180;

// ---------------------------------------------------------------- the sun
// NOAA General Solar Position (Spencer's series). `date` is a JS Date (an absolute instant).
export function sunPosition(date, lat, lon) {
  const start = Date.UTC(date.getUTCFullYear(), 0, 1);
  const dayOfYear = Math.floor((date.getTime() - start) / 864e5);
  const hours = date.getUTCHours() + date.getUTCMinutes() / 60 + date.getUTCSeconds() / 3600;
  const g = (2 * Math.PI / 365) * (dayOfYear + (hours - 12) / 24);
  const eqTime = 229.18 * (0.000075 + 0.001868 * Math.cos(g) - 0.032077 * Math.sin(g)
    - 0.014615 * Math.cos(2 * g) - 0.040849 * Math.sin(2 * g));                 // minutes
  const decl = 0.006918 - 0.399912 * Math.cos(g) + 0.070257 * Math.sin(g) - 0.006758 * Math.cos(2 * g)
    + 0.000907 * Math.sin(2 * g) - 0.002697 * Math.cos(3 * g) + 0.00148 * Math.sin(3 * g);
  const trueSolarMin = hours * 60 + eqTime + 4 * lon;
  const hourAngle = (trueSolarMin / 4 - 180) * DEG;
  const phi = lat * DEG;
  const cosZen = Math.sin(phi) * Math.sin(decl) + Math.cos(phi) * Math.cos(decl) * Math.cos(hourAngle);
  const zen = Math.acos(Math.min(1, Math.max(-1, cosZen)));
  const altitude = 90 - zen / DEG;
  // Azimuth, degrees clockwise from north.
  const az = Math.atan2(Math.sin(hourAngle), Math.cos(hourAngle) * Math.sin(phi) - Math.tan(decl) * Math.cos(phi));
  const azimuth = (az / DEG + 180 + 360) % 360;
  return { altitude, azimuth };
}

// Outdoor illuminance in lux. IES Lighting Handbook daylight formulas:
// clear sky:    E_dn = 127.5 exp(-0.21 / sin a) klx (direct normal sun), E_h,sky = 0.8 + 15.5 sqrt(sin a) klx
// overcast sky: E_h,sky = 0.3 + 21 sin a klx, no direct sun.
// Below the horizon, twilight falls about tenfold per 2.6° (600 lx at sunset, ~3 lx at -6°).
export function daylight(altitude, sky = "clear") {
  if (altitude <= 0) {
    const twilight = 600 * Math.exp(0.88 * altitude) + 0.05;          // + a little city and moon light
    return { sunNormal: 0, skyHorizontal: Math.max(0.05, twilight) };
  }
  const s = Math.sin(altitude * DEG);
  if (sky === "overcast") return { sunNormal: 0, skyHorizontal: (0.3 + 21 * s) * 1000 };
  return { sunNormal: 127500 * Math.exp(-0.21 / Math.max(s, 0.02)), skyHorizontal: (0.8 + 15.5 * Math.sqrt(s)) * 1000 };
}

// Colour of a black body at `kelvin` as linear RGB scaled to luminance 1 (Tanner Helland's fit).
export function kelvinToRGB(kelvin) {
  const t = Math.min(40000, Math.max(1000, kelvin)) / 100;
  let r, g, b;
  if (t <= 66) { r = 255; g = 99.4708025861 * Math.log(t) - 161.1195681661; }
  else { r = 329.698727446 * Math.pow(t - 60, -0.1332047592); g = 288.1221695283 * Math.pow(t - 60, -0.0755148492); }
  if (t >= 66) b = 255; else if (t <= 19) b = 0; else b = 138.5177312231 * Math.log(t - 10) - 305.0447927307;
  const lin = (c) => { c = Math.min(255, Math.max(0, c)) / 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
  const rgb = [lin(r), lin(g), lin(b)];
  const y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
  return rgb.map((c) => c / y);
}

// ---------------------------------------------------------------- lamps
// Typical catalogue figures for bulbs sold today (230 V markets). `life` in hours.
// tech: led (constant-current driver), cfl (ballast), inc (incandescent), hal (mains halogen).
export const BULBS = {
  "led-9":   { name: "LED bulb 9 W",           tech: "led", W: 9,  lm: 900,  cct: 6500, pf: 0.9,  life: 15000, Vr: 230 },
  "led-9w":  { name: "LED bulb 9 W warm",      tech: "led", W: 9,  lm: 806,  cct: 2700, pf: 0.9,  life: 15000, Vr: 230 },
  "led-5":   { name: "LED lamp 5 W",           tech: "led", W: 5,  lm: 450,  cct: 3000, pf: 0.7,  life: 15000, Vr: 230 },
  "tube-20": { name: "LED tube light 20 W",    tech: "led", W: 20, lm: 2000, cct: 6500, pf: 0.95, life: 25000, Vr: 230 },
  "panel-15":{ name: "LED panel 15 W",         tech: "led", W: 15, lm: 1500, cct: 4000, pf: 0.9,  life: 25000, Vr: 230 },
  "cfl-15":  { name: "CFL 15 W",               tech: "cfl", W: 15, lm: 850,  cct: 2700, pf: 0.55, life: 8000,  Vr: 230 },
  "inc-60":  { name: "Incandescent 60 W",      tech: "inc", W: 60, lm: 710,  cct: 2800, pf: 1.0,  life: 1000,  Vr: 230 },
  "hal-42":  { name: "Halogen 42 W",           tech: "hal", W: 42, lm: 630,  cct: 2900, pf: 1.0,  life: 2000,  Vr: 230 },
};

// What a bulb does at supply voltage `V` and dimmer `dim` (0..1). A phase-cut dimmer on a
// filament lamp lowers its RMS voltage, so dimming and a voltage sag follow the same laws:
// flux ~ V^3.4, power ~ V^1.6, colour temperature ~ V^0.42, life ~ V^-13 (standard filament exponents).
// An LED's driver holds the current steady from about 90 to 265 V, so mains sag changes nothing
// until the driver drops out; dimming an LED cuts its flux in proportion, with driver loss on top.
// A CFL's ballast is not dimmable and will not strike below about 170 V.
export function bulbState(bulb, V, dim = 1) {
  const out = { lumens: 0, watts: 0, cct: bulb.cct, amps: 0, va: 0, life: bulb.life, note: "" };
  if (bulb.tech === "inc" || bulb.tech === "hal") {
    const v = (V * dim) / bulb.Vr;
    out.lumens = bulb.lm * Math.pow(v, 3.4);
    out.watts = bulb.W * Math.pow(v, 1.6);
    out.cct = bulb.cct * Math.pow(v, 0.42);
    out.life = bulb.life * Math.pow(Math.max(v, 0.05), -13);
    if (v > 1.05) out.note = `${Math.round((v - 1) * 100)}% over-voltage: life cut to ${Math.round(out.life)} h`;
    else if (dim < 1) out.note = "dimming a filament makes it warmer and far less efficient";
  } else if (bulb.tech === "led") {
    if (V < 90) { out.note = "driver drops out below ~90 V: the lamp flickers or goes dark"; return out; }
    out.lumens = bulb.lm * dim;
    out.watts = bulb.W * (0.1 + 0.9 * dim);
    if (V > 265) out.note = "over-voltage: outside the driver's range";
  } else if (bulb.tech === "cfl") {
    if (V < 170) { out.note = "the ballast will not strike below ~170 V"; return out; }
    const v = V / bulb.Vr;
    out.lumens = bulb.lm * Math.pow(v, 1.2);
    out.watts = bulb.W * Math.pow(v, 1.1);
    if (dim < 1) out.note = "a CFL is not dimmable: the dimmer is ignored";
  }
  out.amps = out.watts / (V * bulb.pf);
  out.va = V * out.amps;
  return out;
}

// Luminous intensity (cd) of a fixture towards a unit direction `d` (from the lamp), given its
// flux and type. Fixtures point down along -y unless `axis` says otherwise.
//   bare:   an open bulb, the same in every direction  I = Φ / 4π
//   panel:  a ceiling panel or batten, Lambertian downward  I = (Φ/π) cos θ
//   shade:  a lamp under a cone shade, uniform within the cone half-angle
export function intensity(fixture, lumens, d, axis = [0, -1, 0], cone = 40) {
  const c = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
  if (fixture === "bare") return lumens / (4 * Math.PI);
  if (fixture === "panel") return c > 0 ? (lumens / Math.PI) * c : 0;
  const cc = Math.cos(cone * DEG);
  // 85% leaves through the cone; the rest is the shade's glow, spread evenly.
  const inCone = (0.85 * lumens) / (2 * Math.PI * (1 - cc));
  return (c >= cc ? inCone : 0) + (0.15 * lumens) / (4 * Math.PI);
}

// ---------------------------------------------------------------- light in the room
// `faces` are rectangles {O, U, V (unit vectors), w, h, n (unit), rho: [r,g,b], owner}.
// `boxes` are the solid pieces that cast shadows, {min, max}. Coordinates: metres, y up.
export class Radiosity {
  constructor(faces, boxes, patchSize = 0.2) {
    this.faces = faces; this.boxes = boxes;
    const P = [], N = [], A = [], F = [], IJ = [];
    faces.forEach((f, fi) => {
      f.nu = Math.max(1, Math.ceil(f.w / patchSize)); f.nv = Math.max(1, Math.ceil(f.h / patchSize));
      f.first = P.length / 3;
      const du = f.w / f.nu, dv = f.h / f.nv;
      for (let j = 0; j < f.nv; j++) for (let i = 0; i < f.nu; i++) {
        const a = (i + 0.5) * du, b = (j + 0.5) * dv;
        for (let k = 0; k < 3; k++) P.push(f.O[k] + f.U[k] * a + f.V[k] * b);
        N.push(...f.n); A.push(du * dv); F.push(fi); IJ.push(i, j);
      }
    });
    this.n = A.length;
    this.pos = new Float32Array(P); this.nrm = new Float32Array(N); this.area = new Float32Array(A);
    this.face = new Int32Array(F); this.ij = new Int32Array(IJ);
    this.direct = new Float32Array(this.n * 3);     // lux arriving straight from lamps, sun and sky
    this.indirect = new Float32Array(this.n * 3);   // lux arriving after one or more bounces
    this.rho = new Float32Array(this.n * 3);
    this.updateReflectance();
    this.formFactors();
  }

  updateReflectance() {
    for (let p = 0; p < this.n; p++) for (let c = 0; c < 3; c++) this.rho[p * 3 + c] = this.faces[this.face[p]].rho?.[c] ?? 0.5;
  }

  // Is the segment a -> b clear of every box? (slab test; ends just off their surfaces)
  clear(ax, ay, az, bx, by, bz) {
    const dx = bx - ax, dy = by - ay, dz = bz - az;
    for (const box of this.boxes) {
      let t0 = 1e-4, t1 = 1 - 1e-4, ok = true;
      for (let k = 0; k < 3 && ok; k++) {
        const o = k === 0 ? ax : k === 1 ? ay : az, d = k === 0 ? dx : k === 1 ? dy : dz;
        if (Math.abs(d) < 1e-12) { if (o <= box.min[k] || o >= box.max[k]) ok = false; continue; }
        let ta = (box.min[k] - o) / d, tb = (box.max[k] - o) / d;
        if (ta > tb) [ta, tb] = [tb, ta];
        if (ta > t0) t0 = ta; if (tb < t1) t1 = tb;
        if (t0 >= t1) ok = false;
      }
      if (ok) return false;
    }
    return true;
  }

  // Point-to-patch form factors F[i][j] = cos θi cos θj A_j / (π r²), with visibility.
  formFactors() {
    const n = this.n, F = new Float32Array(n * n), p = this.pos, q = this.nrm, A = this.area;
    const e = 2e-3;
    for (let i = 0; i < n; i++) {
      const ix = p[i * 3] + q[i * 3] * e, iy = p[i * 3 + 1] + q[i * 3 + 1] * e, iz = p[i * 3 + 2] + q[i * 3 + 2] * e;
      for (let j = i + 1; j < n; j++) {
        if (this.face[i] === this.face[j]) continue;
        const dx = p[j * 3] - p[i * 3], dy = p[j * 3 + 1] - p[i * 3 + 1], dz = p[j * 3 + 2] - p[i * 3 + 2];
        const r2 = dx * dx + dy * dy + dz * dz, r = Math.sqrt(r2);
        const ci = (dx * q[i * 3] + dy * q[i * 3 + 1] + dz * q[i * 3 + 2]) / r;
        const cj = -(dx * q[j * 3] + dy * q[j * 3 + 1] + dz * q[j * 3 + 2]) / r;
        if (ci <= 0 || cj <= 0) continue;
        const jx = p[j * 3] + q[j * 3] * e, jy = p[j * 3 + 1] + q[j * 3 + 1] * e, jz = p[j * 3 + 2] + q[j * 3 + 2] * e;
        if (!this.clear(ix, iy, iz, jx, jy, jz)) continue;
        const g = (ci * cj) / (Math.PI * (r2 + (A[i] + A[j]) / (2 * Math.PI)));   // softened near contact
        F[i * n + j] = g * A[j]; F[j * n + i] = g * A[i];
      }
    }
    for (let i = 0; i < n; i++) {     // nothing sees more than its whole hemisphere
      let s = 0; for (let j = 0; j < n; j++) s += F[i * n + j];
      if (s > 1) for (let j = 0; j < n; j++) F[i * n + j] /= s;
    }
    this.F = F;
  }

  // Direct light at a point with normal nrm. `sources` from lab.js:
  //   lamps: [{pos, fixture, lumens, rgb, axis, cone}]
  //   sun:   {dir (towards the sun), lux, rgb} entering only through `window`
  //   window: {x0, x1, y0, y1, z, luminance (nits, inward), rgb}  — the opening as a glowing panel
  // Returns [lampsRGB, sunRGB, skyRGB].
  directAt(x, y, z, nx, ny, nz, sources) {
    const lamps = [0, 0, 0], sun = [0, 0, 0], sky = [0, 0, 0], e = 2e-3;
    const ox = x + nx * e, oy = y + ny * e, oz = z + nz * e;
    for (const L of sources.lamps) {
      const dx = L.pos[0] - x, dy = L.pos[1] - y, dz = L.pos[2] - z;
      const r2 = dx * dx + dy * dy + dz * dz, r = Math.sqrt(r2);
      const cos = (dx * nx + dy * ny + dz * nz) / r;
      if (cos <= 0 || !this.clear(ox, oy, oz, L.pos[0], L.pos[1], L.pos[2])) continue;
      const I = intensity(L.fixture, L.lumens, [-dx / r, -dy / r, -dz / r], L.axis, L.cone);
      const E = (I * cos) / Math.max(r2, 0.01);
      for (let c = 0; c < 3; c++) lamps[c] += E * L.rgb[c];
    }
    const W = sources.window;
    const S = sources.sun;
    if (S && S.lux > 0 && W) {
      const [sx, sy, sz] = S.dir, cos = sx * nx + sy * ny + sz * nz;
      if (cos > 0 && sz > 1e-3) {
        const t = (W.z - z) / sz, wx = x + sx * t, wy = y + sy * t;
        if (t > 0 && wx > W.x0 && wx < W.x1 && wy > W.y0 && wy < W.y1 && this.clear(ox, oy, oz, wx, wy, W.z)) {
          for (let c = 0; c < 3; c++) sun[c] += S.lux * cos * S.rgb[c];
        }
      }
    }
    if (W && W.luminance > 0) {
      const k = 5, dxw = (W.x1 - W.x0) / k, dyw = (W.y1 - W.y0) / k, a = dxw * dyw;
      for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
        const wx = W.x0 + (i + 0.5) * dxw, wy = W.y0 + (j + 0.5) * dyw;
        const dx = wx - x, dy = wy - y, dz = W.z - z, r2 = dx * dx + dy * dy + dz * dz, r = Math.sqrt(r2);
        const cp = (dx * nx + dy * ny + dz * nz) / r, cw = dz / r;   // window faces -z, into the room
        if (cp <= 0 || cw <= 0 || !this.clear(ox, oy, oz, wx, wy, W.z - 1e-3)) continue;
        const E = (W.luminance * cp * cw * a) / Math.max(r2, a);
        for (let c = 0; c < 3; c++) sky[c] += E * W.rgb[c];
      }
    }
    return [lamps, sun, sky];
  }

  // Light the room: direct on every patch, then bounces until they settle.
  solve(sources, bounces = 10) {
    const n = this.n, p = this.pos, q = this.nrm;
    this.parts = new Float32Array(n * 9);
    for (let i = 0; i < n; i++) {
      const parts = this.directAt(p[i * 3], p[i * 3 + 1], p[i * 3 + 2], q[i * 3], q[i * 3 + 1], q[i * 3 + 2], sources);
      for (let c = 0; c < 3; c++) {
        this.direct[i * 3 + c] = parts[0][c] + parts[1][c] + parts[2][c];
        this.parts[i * 9 + c] = parts[0][c]; this.parts[i * 9 + 3 + c] = parts[1][c]; this.parts[i * 9 + 6 + c] = parts[2][c];
      }
    }
    const B = new Float32Array(n * 3), F = this.F, ind = this.indirect;
    ind.fill(0);
    for (let it = 0; it < bounces; it++) {
      for (let j = 0; j < n * 3; j++) B[j] = this.rho[j] * (this.direct[j] + ind[j]);
      for (let i = 0; i < n; i++) {
        let r = 0, g = 0, b = 0; const row = i * n;
        for (let j = 0; j < n; j++) { const f = F[row + j]; if (f) { r += f * B[j * 3]; g += f * B[j * 3 + 1]; b += f * B[j * 3 + 2]; } }
        ind[i * 3] = r; ind[i * 3 + 1] = g; ind[i * 3 + 2] = b;
      }
    }
    this.sources = sources;
  }

  // Lux at any point and facing: direct parts plus the bounced light from every patch.
  luxAt(x, y, z, nx, ny, nz) {
    const [lamps, sun, sky] = this.directAt(x, y, z, nx, ny, nz, this.sources);
    const n = this.n, p = this.pos, q = this.nrm, e = 2e-3, bounced = [0, 0, 0];
    for (let j = 0; j < n; j++) {
      const dx = p[j * 3] - x, dy = p[j * 3 + 1] - y, dz = p[j * 3 + 2] - z;
      const r2 = dx * dx + dy * dy + dz * dz, r = Math.sqrt(r2);
      const ci = (dx * nx + dy * ny + dz * nz) / r, cj = -(dx * q[j * 3] + dy * q[j * 3 + 1] + dz * q[j * 3 + 2]) / r;
      if (ci <= 0 || cj <= 0 || r < 1e-3) continue;
      if (!this.clear(x + nx * e, y + ny * e, z + nz * e, p[j * 3] + q[j * 3] * e, p[j * 3 + 1] + q[j * 3 + 1] * e, p[j * 3 + 2] + q[j * 3 + 2] * e)) continue;
      const f = (ci * cj * this.area[j]) / (Math.PI * (r2 + this.area[j] / Math.PI));
      for (let c = 0; c < 3; c++) bounced[c] += f * this.rho[j * 3 + c] * (this.direct[j * 3 + c] + this.indirect[j * 3 + c]);
    }
    const Y = (v) => 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
    return { lamps: Y(lamps), sun: Y(sun), sky: Y(sky), bounced: Y(bounced) };
  }

  // Area-weighted log-average luminance of the room's surfaces (nits): what the eye adapts to.
  adaptation() {
    let s = 0, a = 0;
    for (let i = 0; i < this.n; i++) {
      let L = 0;
      for (let c = 0; c < 3; c++) L += [0.2126, 0.7152, 0.0722][c] * this.rho[i * 3 + c] * (this.direct[i * 3 + c] + this.indirect[i * 3 + c]);
      s += this.area[i] * Math.log(L / Math.PI + 1e-3); a += this.area[i];
    }
    return Math.exp(s / a);
  }
}

// ---------------------------------------------------------------- glare
// CIE Unified Glare Rating for an observer at `eye` looking along `look` (unit), `up` (unit).
// lamps: [{pos, luminance (nits of the visible emitting area), area (m², seen face-on)}].
// Lb: background luminance (nits). Guth position index from the Luckiesh–Guth fit.
export function ugr(eye, look, up, lamps, Lb) {
  let sum = 0;
  const right = [look[1] * up[2] - look[2] * up[1], look[2] * up[0] - look[0] * up[2], look[0] * up[1] - look[1] * up[0]];
  for (const L of lamps) {
    const d = [L.pos[0] - eye[0], L.pos[1] - eye[1], L.pos[2] - eye[2]];
    const r = Math.hypot(...d), u = d.map((v) => v / r);
    const along = u[0] * look[0] + u[1] * look[1] + u[2] * look[2];
    if (along <= 0.05) continue;                                       // behind or at the edge of view
    const beta = Math.acos(Math.min(1, along)) / DEG;
    const x = u[0] * right[0] + u[1] * right[1] + u[2] * right[2], y = u[0] * up[0] + u[1] * up[1] + u[2] * up[2];
    const alpha = (Math.atan2(Math.abs(x), Math.abs(y) + 1e-9)) / DEG;   // from the vertical through the line of sight
    const lnP = (35.2 - 0.31889 * alpha - 1.22 * Math.exp((-2 * alpha) / 9)) * 1e-3 * beta
      + (21 + 0.26667 * alpha - 0.002963 * alpha * alpha) * 1e-5 * beta * beta;
    const P = Math.exp(lnP), omega = L.area / (r * r);
    sum += (L.luminance * L.luminance * omega) / (P * P);
  }
  if (sum === 0) return null;
  return 8 * Math.log10((0.25 / Math.max(Lb, 1)) * sum);
}

export function ugrWords(v) {
  if (v == null) return "no lamp in view";
  if (v < 13) return "no discomfort";
  if (v < 19) return "acceptable for reading and desk work (limit 19)";
  if (v < 22) return "noticeable: above the desk-work limit";
  if (v < 28) return "uncomfortable";
  return "intolerable";
}
