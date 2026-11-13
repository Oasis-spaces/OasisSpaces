# Realism: light, shadow, materials, energy (November 2026)

*Assume the space itself is perfect: geometry and every surface's look are right. This
note is about what makes it look and behave like the real room: daylight and the sun,
shadows, what each thing is made of, glare, and lamps that can be added, switched and
dimmed, with what they draw from the mains.*

## The core problem

A splat (or a photo texture) stores **the light that was there when it was filmed**. It
is a picture of the room under one lighting, not a description of the room. That is why
it looks right from the filmed views and wrong as soon as anything changes: switch a lamp
off and the lamp's glow stays painted on the wall; add one and nothing gets brighter
around it. The phrase "looks animated" came from the opposite mistake, which was to throw
the photo away and paint flat colours lit by made-up lights.

To get both realism and editable light, the room has to be split into:

1. **what each surface is**: its own colour (albedo), how glossy it is (roughness), which
   way it faces (normal), and whether it glows;
2. **what light there is**: lamps with real output, the sun and sky through the openings,
   and light bouncing between surfaces;

and then put back together by a renderer that obeys physics. When it is done right, the
recombined picture under the filmed lighting matches the video. That check is the test,
and it is measurable on held-out frames, as the splat work already does.

## What exists now: `lighting-lab/`

The lab works on the hand-built bedroom (`handbuilt-room/`: its box geometry, every surface
photographed from IMG_4182) and does the whole chain in the browser, in SI units:

| Part | How | Where |
|---|---|---|
| Removing the filmed light | Divide out each photo's broad light and shade (a blur a sixth of the surface wide). The pattern and the average colour stay. This is a classical first estimate of albedo, and the toggle shows the difference. | `lab.js` `photo()` |
| Sun | NOAA solar position for latitude, longitude, date and time. Direct-normal and sky illuminance from the IES clear and overcast formulas, with twilight below the horizon. Sunlight enters only through the window opening (glass τ 0.8), with real shadows. | `physics.js` `sunPosition`, `daylight` |
| Sky | The opening glows with the sky's luminance. Drawn curtains glow with what passes through them (τ ≈ 0.12), tinted by their own photographed colour. | `lab.js` `relight()` |
| Lamps | Catalogue bulbs: LED, LED tube, LED panel, CFL, incandescent and halogen, each with lumens, watts, colour temperature, power factor and life. Fixtures: bare bulb (isotropic), ceiling panel (Lambertian), shaded table lamp (cone). Each can be placed by clicking a surface, switched, dimmed and swapped. | `physics.js` `BULBS`, `intensity` |
| Voltage | A filament follows flux ∝ V³·⁴, power ∝ V¹·⁶, colour temperature ∝ V⁰·⁴², life ∝ V⁻¹³. A phase dimmer acts the same way, which is why dimmed filaments go orange. An LED driver holds steady from about 90 to 265 V and then drops out. A CFL won't strike below about 170 V and can't be dimmed. | `physics.js` `bulbState` |
| Energy | Watts, amps (P / V·PF), VA, kWh a month at each lamp's hours, cost at the tariff, CO₂ at the grid factor, the heat added to the room, and what swapping to LED would save. | `lab.js` `report()` |
| Bounced light | Radiosity on about 2,000 patches of 20 cm, with form factors and visibility against the furniture. It is drawn as light maps. The direct light of point-like sources is drawn by three.js with shadow maps, so their shadows stay sharp. Panels and the sky, being broad, go in the light maps. | `physics.js` `Radiosity` |
| Lux meter | Click any surface for its lux, split into lamps, sun, sky and bounced light. It is checked against the recommended level for the task: 500 lx at the desk, 300 lx for reading in bed, 100 lx general (EN 12464-1 / IES). A false-colour lux map is also available. | `luxAt`, `need()` |
| Materials | Each surface has a material (matt paint, glazed tile, laminate, wood, fabric with sheen, glass screen), a gloss slider, and a lighter or darker control. Its reflectance comes from its photograph. | `lab.js` `materialControls` |
| Eye and glare | Exposure is metered from the view, the way an eye or camera adapts (log-average, top 2% ignored, about 0.5 s to adapt, not fully in the dark). Bloom appears on things far brighter than the adapted level. Discomfort glare from lamps in view is given as CIE UGR, with background luminance taken as the indirect illuminance at the eye ÷ π. A warning shows when the sun's disc or a window far brighter than the room is in view. | `ugr`, `meter()` |

Checked numbers, all at 230 V:

- **Ceiling panel at night:** a 1,500 lm ceiling panel gives about 80 lx on the floor.
- **Desk lamp:** the 450 lm desk lamp gives about 890 lx on the desk under it.
- **Reading in bed:** at night the bed gets 122 lx, which is 41% of the 300 lx needed to read.
- **Sunny 11:00 (sun 41° up, south window, Delhi):** the floor in the sun patch gets about 49 klx. The rest of the room gets 1–4 klx of bounced light.
- **Curtains closed at 11:00:** the room drops to 250–480 lx.
- **Mains sag:** at 200 V a 60 W incandescent gives 441 lm at 48 W and 2,640 K.

What the lab does **not** claim:

- The window is assumed to be where the curtains hang.
- The two starting lamps are placeholders, because the video's real fittings are not detected yet.
- The place, time and compass direction are editable placeholders.
- The floor and the sides of the bed were barely filmed, so they keep plain colours.

## Getting there from the real capture

Each step reuses what the pipeline already has.

1. **Absolute light from the phone.** Every frame has ISO, shutter and aperture. The
   reflected-light meter equation (N²/t = L·S/K, K ≈ 12.5) turns pixel values into
   cd/m². `splat_gsplat.py` already corrects exposure per frame, so the
   splat can be trained in **absolute luminance** instead of per-frame relative colour.
   That is what lets the room be measured: lux on the desk as filmed, not guessed.
   The capture app should record EXIF/AVFoundation exposure for each frame (and lock it
   when it can).
2. **Find the lights that were on.** Lamps are small saturated regions that stay bright
   from every view (detector names plus the cloud's position). Windows are already found
   by `pipeline/mirrors.py`. Each gets a position and size from the cloud. Its output
   (lumens) is **fitted** so that radiosity under those sources reproduces the measured
   wall and ceiling luminances. That gives a light model checked against the video, not
   invented.
3. **Split albedo from shading.** Do it per frame, then fuse it into the splat:
   - **Shading:** solved from the fitted light model and the geometry (radiosity, as in the lab).
   - **Albedo:** albedo = measured radiance × π / irradiance.
   - **Learned priors:** where a learned intrinsic-image model helps (fine shadows,
     specular highlights), it can be checked against the physics. Candidates to evaluate,
     after checking licence and versions: Marigold-IID (appearance and lighting
     variants), RGB↔X, DiffusionRenderer. Relightable-Gaussian training (GS-IR, R3DG,
     IRGS) puts albedo, roughness and a normal on each splat directly, with MoGe depth
     as the normal prior we already train with.
4. **Relight the real splat in the viewer.** The same radiosity fills an
   **irradiance probe grid** (a coarse 3D texture over the room). Each splat's colour
   becomes albedo × irradiance / π, plus a specular term from its roughness, evaluated in
   Spark's splat shader. Direct light from point-like lamps and the sun adds shadow-map
   visibility. With the filmed lights at their fitted output, the result must match the
   held-out frames (PSNR/SSIM, as now). It can then be changed freely.
5. **Surfaces for light to land on.** Radiosity needs surfaces, not splats. Use the
   room's planes (already found) and object boxes or meshes as the light-transport
   proxy, as the lab does with the hand-built boxes. The splat stays the visible surface.
6. **Outside.** A window's view comes from frames that looked out through it. If no frame
   did, it stays a plain panel of sky luminance and is never invented. The orientation
   comes from the phone's compass heading recorded at capture, and the location comes
   from the user (not inferred).

## Order of work

1. Add per-frame exposure metadata to the capture and to `splat_gsplat.py`, so splats train in cd/m².
2. Detect lamps and windows and fit their lumens on the bedroom. Report measured against fitted lux at a few spots.
3. Build an irradiance probe grid from the room proxy, and an albedo splat (divide by the fitted irradiance). Score it on held-out frames under the fitted lights.
4. Add Spark shader relighting in `scene-viewer`, using the lab's lamp, sun, energy and glare panel as is.
5. Add learned intrinsic and relightable-Gaussian methods only where step 3's residuals show the physics alone falls short (fine contact shadows, gloss).
