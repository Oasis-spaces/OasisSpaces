// Drives scene-viewer/ in headless Chromium for tools/tests/test_scene_viewer.py.
//   node scene_viewer_check.mjs <base url> <folder with chair.glb, lamp-cm.glb>
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const [base, folder] = process.argv.slice(2);
const page_url = `${base}/scene-viewer/index.html?scene=../scene/scene.json`;

const failures = [];
function check(ok, what) {
  console.log(`${ok ? "ok  " : "FAIL"} ${what}`);
  if (!ok) failures.push(what);
}
const near = (a, b, tolerance = 0.02) => Math.abs(a - b) <= tolerance;

const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1100, height: 750 } });
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });

async function open(url) {
  await page.goto(url);
  await page.waitForFunction(() => window.oasisScene?.ready(), null, { timeout: 60000 });
  // Every splat loaded, every model in, the loading screen gone and a few frames drawn.
  await page.evaluate(async () => {
    for (const p of window.oasisScene.pieces.values()) { await p.splat?.initialized; await p.modelReady; }
  });
  await page.waitForTimeout(1500);
}
await open(page_url);
check(await page.evaluate(() => window.oasisScene.pieces.size) === 3, "the scene's three pieces load");

// ------------------------------------------------------------- generated models
const generated = await page.evaluate(() => {
  const { pieces } = window.oasisScene;
  const bed = pieces.get("B0");
  bed.group.updateMatrixWorld(true);
  const box = new bed.group.position.constructor();
  const bounds = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  bed.generated.traverse((node) => {
    if (!node.isMesh) return;
    const position = node.geometry.attributes.position;
    for (let i = 0; i < position.count; i++) {
      box.fromBufferAttribute(position, i).applyMatrix4(node.matrixWorld);
      box.toArray().forEach((v, k) => { bounds.min[k] = Math.min(bounds.min[k], v); bounds.max[k] = Math.max(bounds.max[k], v); });
    }
  });
  const order = [];
  window.oasisScene.select("B0");
  for (let i = 0; i < 3; i++) { document.getElementById("swap").click(); order.push(bed.showing); }
  return { bounds, looks: Object.keys(bed.looks), order, visible: bed.generated.visible };
});
check(generated.looks.join() === "scan,model,generated", "the bed has a scan, a clean model and a generated model");
check(generated.bounds.min.every((v, k) => near(v, [0.5 - 0.7, 0, 0.2 - 1.0][k])) &&
      generated.bounds.max.every((v, k) => near(v, [0.5 + 0.7, 0.5, 0.2 + 1.0][k])),
      `the generated model, turned and scaled as recorded, fills the bed's box (${JSON.stringify(generated.bounds)})`);
check(generated.order.join() === "model,generated,scan", `the swap button goes through all three (${generated.order})`);
const sources = await page.evaluate(() => {
  const out = [];
  window.oasisScene.select("B0");
  for (let i = 0; i < 3; i++) { document.getElementById("swap").click(); out.push(document.getElementById("sel-source").textContent); }
  return out;
});
check(sources[0].startsWith("Clean model") && sources[1].startsWith("Generated") && sources[2] === "As filmed",
      `the selected piece says where each look comes from (${sources.join(" | ")})`);
const filledLine = await page.textContent("#filled");
check(filledLine.includes("73% filmed") && filledLine.includes("27% filled in by inpainting"),
      `the panel says how much of the room's surfaces was filmed (${filledLine})`);
await page.evaluate(() => window.oasisScene.select(null));

// ------------------------------------------------------------- filled areas
// The floor's right half (+x) was invented: tinted magenta when filled areas show, its left half untouched.
async function floorPixels() {
  const at = await page.evaluate(() => {
    const { camera } = window.oasisScene;
    const project = (x, z) => { const v = camera.position.clone().set(x, 0.001, z).project(camera); return [Math.round((v.x + 1) / 2 * innerWidth), Math.round((1 - v.y) / 2 * innerHeight)]; };
    return [project(1.6, 1.2), project(-1.6, 1.2)];
  });
  await page.waitForTimeout(500);
  const shot = (await page.screenshot()).toString("base64");
  return page.evaluate(async ([data, pts]) => {
    const image = new Image(); image.src = `data:image/png;base64,${data}`; await image.decode();
    const canvas = document.createElement("canvas"); canvas.width = image.width; canvas.height = image.height;
    const context = canvas.getContext("2d"); context.drawImage(image, 0, 0);
    return pts.map(([x, y]) => Array.from(context.getImageData(x, y, 1, 1).data.slice(0, 3)));
  }, [shot, at]);
}
const plainFloor = await floorPixels();
await page.click("#show-filled");
await page.waitForFunction(() => [...window.oasisScene.shellMeshes.values()].some((m) => m.userData.overlay && m.userData.overlay.visible));
const tintedFloor = await floorPixels();
const redder = (a, b) => (a[0] - a[1]) - (b[0] - b[1]);
check(redder(tintedFloor[0], plainFloor[0]) > 20 && Math.abs(redder(tintedFloor[1], plainFloor[1])) < 8,
      `Show filled areas tints the invented half of the floor and leaves the filmed half (${plainFloor} -> ${tintedFloor})`);
await page.click("#show-filled");
check((await page.textContent("#show-filled")) === "Show filled areas", "and hides it again");

// ------------------------------------------------------------- colours
// The colour on screen where the bed is: the screenshot read back in the page.
async function bedPixel() {
  const at = await page.evaluate(() => {
    const { pieces, camera } = window.oasisScene;
    const centre = pieces.get("B0").group.localToWorld(pieces.get("B0").group.position.clone().set(0, 0.5, 0));
    const ndc = centre.project(camera);
    return [Math.round((ndc.x + 1) / 2 * innerWidth), Math.round((1 - ndc.y) / 2 * innerHeight)];
  });
  await page.waitForTimeout(800);                         // a few frames, for the splat's new colours
  const shot = (await page.screenshot()).toString("base64");
  return page.evaluate(async ([data, x, y]) => {
    const image = new Image();
    image.src = `data:image/png;base64,${data}`;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.width; canvas.height = image.height;
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0);
    const px = context.getImageData(x - 3, y - 3, 7, 7).data;
    const mean = [0, 0, 0];
    for (let i = 0; i < px.length; i += 4) for (let c = 0; c < 3; c++) mean[c] += px[i + c] / 49;
    return mean.map(Math.round);
  }, [shot, ...at]);
}
const before = await bedPixel();
const tinted = await page.evaluate(async () => {
  const { pieces, setTint } = window.oasisScene;
  await setTint("B0", "#3366cc");
  const bed = pieces.get("B0");
  const material = bed.model.getObjectByProperty("isMesh", true).material;
  const mean = [0, 0, 0];
  bed.splat.packedSplats.forEachSplat((i, c, s, q, o, color) => { mean[0] += color.r; mean[1] += color.g; mean[2] += color.b; });
  return { splat: mean.map((v) => v / bed.splat.packedSplats.numSplats), model: material.color.toArray(),
           row: bed.row.querySelector(".swatch").style.background };
});
const after = await bedPixel();
// #3366cc is (0.2, 0.4, 0.8): the splat's average comes out near it, at its own brightness.
check(near(tinted.splat[0], 0.2, 0.05) && near(tinted.splat[2], 0.8, 0.1),
      `the bed's splat takes the colour (average ${tinted.splat.map((v) => v.toFixed(2))})`);
check(before[0] > before[2] && after[2] > after[0], `on screen it turns from red to blue (${before} -> ${after})`);
check(tinted.model[2] > tinted.model[0], "its clean model turns blue too");
check(tinted.row !== "", "its row shows the colour");
const untinted = await page.evaluate(async () => {
  const { pieces, setTint } = window.oasisScene;
  await setTint("B0", null);
  const bed = pieces.get("B0");
  const material = bed.model.getObjectByProperty("isMesh", true).material;
  let same = true;
  bed.splat.packedSplats.forEachSplat((i, c, s, q, o, color) => {
    same &&= Math.abs(color.r - bed.own.splat[3 * i]) < 0.01 && Math.abs(color.b - bed.own.splat[3 * i + 2]) < 0.01;
  });
  return { same, model: material.color.equals(material.userData.originalColor) };
});
check(untinted.same && untinted.model, "its own colours come back");
check((await bedPixel()).every((v, i) => Math.abs(v - before[i]) < 12), "and on screen too");

// ------------------------------------------------------------- added objects
await page.setInputFiles("#picker", `${folder}/chair.glb`);
await page.waitForFunction(() => window.oasisScene.pieces.has("added-1"));
const chair = await page.evaluate(() => {
  const piece = window.oasisScene.pieces.get("added-1");
  const THREE_box = piece.info.box;
  return { label: piece.info.label, box: THREE_box, inAdded: !!piece.row.closest("#added"),
           selected: piece.row.classList.contains("selected"), height: document.getElementById("height").value,
           heightShown: !document.getElementById("height-wrap").hidden };
});
check(chair.label === "chair" && chair.inAdded, "a GLB from the file picker is added as a piece, listed under Added");
check(chair.box.min[1] === 0 && near(chair.box.max[1], 0.9) && near(chair.box.max[0] - chair.box.min[0], 0.5),
      `it stands on the floor at its own size in metres (${JSON.stringify(chair.box)})`);
check(chair.selected && chair.heightShown && chair.height === "90", "it is selected, with its height in centimetres");

await page.fill("#height", "45");
await page.dispatchEvent("#height", "change");
const half = await page.evaluate(() => window.oasisScene.pieces.get("added-1").group.scale.x);
check(near(half, 0.5, 1e-6), "typing a height scales it evenly");

await page.setInputFiles("#picker", `${folder}/lamp-cm.glb`);
await page.waitForFunction(() => window.oasisScene.pieces.has("added-2"));
const lamp = await page.evaluate(() => window.oasisScene.pieces.get("added-2").info.box);
check(near(lamp.max[1], 1.6) && near(lamp.max[0] - lamp.min[0], 0.3), "a model made in centimetres is read as centimetres");

// ------------------------------------------------------------- layouts
const saved = await page.evaluate(async () => {
  const { pieces, setTint, currentLayout } = window.oasisScene;
  const bed = pieces.get("B0"), chair = pieces.get("added-1");
  bed.group.position.set(-0.8, 0, 0.4);
  bed.group.rotation.y = Math.PI / 2;
  chair.group.position.set(1.2, 0, -0.9);
  await setTint("B0", "#228844");
  await setTint("added-1", "#aa3311");
  pieces.get("added-2").group.visible = false;
  for (const p of pieces.values()) p.group.updateMatrixWorld(true);
  return currentLayout();
});
check(saved.added.length === 2 && saved.added.every((a) => a.glb.startsWith("data:model/gltf-binary;base64,")),
      "the saved layout carries both added objects whole");
const pillowBefore = saved.pieces.find((p) => p.id === "B1").position;

await open(page_url);                                     // a fresh room
const restored = await page.evaluate(async (layout) => {
  const { pieces, applyLayout } = window.oasisScene;
  await applyLayout(layout);
  const world = (id) => pieces.get(id).group.getWorldPosition(pieces.get(id).group.position.clone()).toArray();
  return { ids: [...pieces.keys()], bed: world("B0"), bedTurn: pieces.get("B0").group.rotation.y,
           pillow: world("B1"), chair: world("added-1"), chairScale: pieces.get("added-1").group.scale.x,
           bedTint: pieces.get("B0").tint, chairTint: pieces.get("added-1").tint,
           lampHidden: !pieces.get("added-2").group.visible };
}, saved);
check(restored.ids.includes("added-1") && restored.ids.includes("added-2"), "loading the layout brings the added objects back");
check(near(restored.bed[0], -0.8) && near(restored.bed[2], 0.4) && near(restored.bedTurn, Math.PI / 2), "the bed is where it was left, turned");
check(restored.pillow.every((v, i) => near(v, pillowBefore[i])), "the pillow it carries is where it was left too");
check(near(restored.chair[0], 1.2) && near(restored.chairScale, 0.5, 1e-6), "the chair keeps its place and its size");
check(restored.bedTint === "#228844" && restored.chairTint === "#aa3311", "colours come back");
check(restored.lampHidden, "a hidden piece stays hidden");

// The same through the URL, as the apps open a saved room.
const layoutFile = `${folder}/layout.json`;
(await import("node:fs")).writeFileSync(layoutFile, JSON.stringify(saved));
await open(`${page_url}&layout=../layout.json`);
await page.waitForFunction(() => window.oasisScene.pieces.has("added-1") && window.oasisScene.pieces.get("B0").tint);
check(true, "?layout= opens a saved layout");
await page.evaluate(() => window.oasisScene.select("B0"));
await page.screenshot({ path: process.env.SCENE_VIEWER_SHOT || `${folder}/viewer.png` });
console.log(`screenshot: ${folder}/viewer.png`);

check(errors.length === 0, `no errors in the page${errors.length ? ": " + errors.join(" | ") : ""}`);
await browser.close();
process.exit(failures.length ? 1 : 0);
