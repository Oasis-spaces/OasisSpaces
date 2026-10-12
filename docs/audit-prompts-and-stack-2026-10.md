# Prompts and tech stack against 16 clips and the field (October 2026)

Done on 11-12 October 2026 on the cloud box (4 CPU cores, 15 GB, no GPU, Hugging Face blocked), on the
16 clips of the `test-videos` release: ten from the earlier cross-clip study and six new ones (an
open-plan studio filmed twice, a hotel suite seen through an arch, a bedroom with a lot of ceiling,
a 23-second walkthrough of a living room and a bedroom, and a close slide past a bedside). The
question: are the prompts universal, does each say where it sits and how it helps the
reconstruction, and has every piece of technology, in the capture app and in the pipeline, earned
its place?

Three kinds of evidence, kept apart below:

- **Measured here:** stage 1 of the pipeline on all 16 clips; a controlled COLMAP bench (the same
  frames through each mapper, feature type, calibration and sampling option, the incremental mapper
  over three seeds); repeated fresh runs of the two hardest clips; and the four Claude questions that
  need only stage 1, asked on all 16 clips with the old prompts and the new, then re-asked after
  each fix until the misreadings stopped.
- **Read:** COLMAP's changelog and FAQ, model READMEs and licences, GitHub issues, Anthropic's
  prompting and vision docs, papers, Hacker News. Reddit, X and LinkedIn cannot be fetched by
  Firecrawl, so their threads are quoted from search excerpts only.
- **Not testable here:** everything that needs a GPU or Hugging Face weights (MoGe-2, GroundingDINO,
  SAM 2.1, OpenSplat training, SAM 3, Depth Anything 3), and the capture app (no Swift toolchain).
  Those verdicts rest on the earlier T4 run's numbers and the reading, and are put as decisions,
  not changes.

## In short

- **The prompts** now say what the pipeline makes, where each question sits and what its answer
  changes, and name the capture as it is (a video or a set of photos, one room or several). On the
  same 16 clips the capture verdict stopped asking to retry solves that had placed every frame (9
  of 15 before, 0 after), and the end-of-run advice stopped reading stages that never ran as
  failures (10 before, 0 after). Two misreadings survived the wording and were fixed by changing
  what the advisor is given instead: sizes measured on a guessed scale (one room came out 95 m
  high) and a frame rate it turned into the wrong clip length.
- **Stage 1** had a bug and a weak fallback. A global solve that placed frames but no point was
  accepted, so the slow studio slide stopped with "no parallax". The fallback now runs view-graph
  calibration, then the incremental mapper with small-baseline triangulation angles and COLMAP's
  own, three seeds each. Over three seeds on all 16 clips the small-baseline angles placed more
  frames than COLMAP's in 22 runs and fewer in 2. The two clips that used to stop at stage 1 now
  solve on 4 of 4 and 3 of 4 fresh runs.
- **Removed from the retries:** ALIKED features (never placed more frames than SIFT, fewer points on
  all 14 clips both solved). **Measured and not adopted:** LoMa, DEGENSAC, view-graph calibration as a first
  step, AnyCalib, and three times the frames as the usual retry.
- **One correction.** A first, single-run pass of the bench credited the small-baseline angles with
  solving the white-bedroom slide; it was one lucky draw of a random solve, it reached a commit,
  and the next commit corrected it. The numbers below are all over seeds or repeated runs.
- **The capture app** now holds exposure and white balance while recording (untested on a device).
  Both hard clips are 6-7-second slides that barely move, and they still sit on a knife edge (the
  matching's small run-to-run differences decide them). The app's existing rules (too short,
  turning on the spot) would have flagged them; a live "the view is not shifting" rule is
  proposed below.

## The prompts: what failed on real clips, and what changed

Method: the four questions that need only stage 1 (the capture verdict, the frame choice, the object
naming and the end-of-run capture advice) were asked on all 16 clips twice, with the prompts as they
were (commit 3fadc23, from a separate worktree) and as they are now, through the same `claude` CLI
backend the pipeline uses. The capture verdict is normally asked only on a weak solve and the advice
only at the end of a run; asking both on every clip is what exposes how they read their inputs. The
other ten questions need stage 2-4 outputs that no GPU here can produce, so they were rewritten from
the failure classes found here, the research, and the code that consumes their answers.

### Failure classes found on the clips

1. **Coverage read as a bad solve.** The old capture verdict asked for a denser retry on clips whose
   every frame was placed in one model (8580864: 33/33; 11299294: 17/17; 38413578: 16/16;
   29814959: 9/9 at 0.37 px; 32144167: "All 25 frames registered cleanly into one model, but the
   sampling is too sparse to cover a room this size"). A retry re-solves the same footage; it cannot
   add a floor that was never filmed. The prompt never said so, and called retry_denser "the first
   thing to try when few frames were kept".
2. **Guessed metres read as measurements.** Before densify the capture map's scale is a guess (the
   phone held 1.4 m up). The advice called correct solves "geometrically degenerate: an 11.49 m walk
   produced a 30 m x 45 m x 18 m box" (29814959) and "physically impossible ... 54.6 m by 33.8 m"
   (38413578), and judged a 23-second walkthrough to be "a single standing viewpoint" from a guessed
   1.11 m path (35308496).
3. **Sentinels read as facts.** `path_jump` was 0.0 both for a still camera and for a model too thin
   to judge; on the low-parallax studio clip (27975843) the advice reported "cameras sit at
   effectively the same viewpoint — zero recorded motion". A tiny leftover component from COLMAP 4.2's
   multi-model global mapper (5 frames, 2 points) turned "18/18 frames" into "the best of 2 models",
   which the old verdict prompt defined as "the capture broke into pieces".
4. **Stages never run read as failures.** With depth, structure and walls empty, every forced advice
   said "the run completed but produced no wall, structure or depth measurements whatsoever".
5. **Causes invented without evidence.** "no switching between wide and main camera mid-capture"
   (37674124): nothing measured supports it.
6. **Video-only and single-room wording.** Every prompt said "a phone video of a room";
   reconstruct.py also takes a folder of photos, and the clips include a multi-room walkthrough
   (35308496), an open-plan studio (27975843/27975847), a suite seen through an arch (29681899) and a
   close bedside detail (7749088). The naming rule "this room only ... seen through a doorway into
   another room gets elsewhere" would send a walked-into bedroom to "elsewhere".
7. **Parts named as objects.** "headboard" was named as furniture of its own (34208854).

And one code bug the probe exposed, worse than any wording: **the frame choice never showed the end
of a 48-frame capture.** `images[::round(48/36)][:36]` kept frames 1-36, so the whole bedroom of the
walkthrough (frames 37-48) reached no question, and the object list had no bed.

### What changed, prompt by prompt

| Prompt | Changes |
|---|---|
| all | Each opens with where it sits (what the pipeline makes, from what) and what its answer changes; names the actual capture ("a phone video" or "a set of photos"); "room" became "space" where a capture can span rooms or open plan. |
| capture verdict | Reading guide for every metric (frames of total is what matters; leftover models; path_jump null; rejected_global and why); "A retry ... cannot add views the capture never took ... Judge the solve, not the coverage"; the capture map is read for path coherence only. The options follow the stage-1 bench: the usual retry is the incremental mapper with small-baseline angles; three times the frames only for a video whose frames show fast turns; the ALIKED retry is gone. |
| walk check | A pan from one spot has a cluster for a path; added frames belong in it. |
| frame choice | Spread first to last (code); cover every room the camera enters; fitted units and appliances in the objects set; fewer frames than asked for a single close view. |
| object naming | Every room the camera enters counts; "elsewhere" only for spaces never entered; whole objects not parts; people and pets as "loose"; an empty space is a valid answer; kitchens, bathrooms and offices (island, bathtub, toilet as furniture; fitted units and vanities as storage; sinks and wall cabinets as fixtures); English names whatever the labels' language. |
| label review | Its revision is the last detection run; same multi-room, people and appliance rules. |
| structure review | Its edit groups are measured and undone when the fit worsens, so it judges what points cannot (what a thing is); open-plan, L-shaped and multi-room spaces have openings, and a missing wall is no reason to drop or add. |
| render check | A structural problem sends the room to a second review and the better-fitting room is kept; open plan is not a missing wall; "advice" must name the wall or area to film for a structural problem, never a general tip. |
| fill review, start view, splat choice, mixed scene | Capture-neutral wording; what keeping or rejecting does (a kept fill ships; the first-ranked splat is published unless it measures clearly worse); the main room of a multi-room capture opens first; an empty room's scene review answers empty pieces. |
| capture advice | Reading guide (leftover models; path_jump is a teleport check, not distance travelled; rejected_global and why; the coverage describes the placed frames; a sparse point count alone is no finding); stages not reached are passed as "not run"; "a cause these numbers do not show is not a finding". Its inputs changed too: the capture map's sizes and floor shares are withheld while its scale is a guess, and sampling gives the seconds of video spanned instead of the candidates' frame rate (see below). |

`tools/show_prompts.py` now lists all 14 asks (the walk check and "where does a dropped piece stand"
were missing) and marks "otherwise" branches.

### Before and after, on the same 16 clips

The same four questions, the same stage-1 spaces, the same `claude` CLI backend; the old prompts
from a worktree of commit 3fadc23. A comparison script flags each misreading class by pattern, and
every flag was read by hand. The old flow crashed on 27975843 (the zero-point solve), so "before"
has 15 clips.

| | Before (15 clips) | After (16 clips) |
|---|---|---|
| Capture verdict: retry on a solve that had placed every frame | 9 | 0 |
| Capture verdict: retry on a genuinely weak solve | 1 (34208854) | 2 (34208854, 27975843) |
| Advice read a stage that never ran as a failure of the capture | 10 | 0 |
| Advice reasoned from guessed metres | 4 | 1, then 0 (see below) |
| Advice named a cause nothing measured (lens switch, lighting) | 2 | 0 |
| Advice read an unjudged path as a camera that never moved | 0 (the old flow had no null) | 1, then 0 |
| Advice abstained | 1 (3769951) | 1 (3769951) |

Checked against the frames by eye:

- **Near-empty bedroom (35023423):** the old naming added a "nightstand"; the frames show only a
  shelf fixed to the headboard. The new answer names bed, wardrobe, curtain, mirror, window, door
  and chandelier, and nothing else.
- **Two-room walkthrough (35308496):** the new frame choice reaches the bedroom (frames 39-47) and
  the naming lists bed, nightstand and dressing table; before, no question saw frames 37-48.
- **Suite through an arch (29681899):** bed, sofa, dining set, stools, wardrobe and rug are all in
  the frames and named; the bed beyond the arch is part of one open space COLMAP placed whole, so it
  is correctly not "elsewhere".
- **White bedroom slide (34208854):** the new advice saw that the capture map was drawn from the
  only 2 placed frames, so its tiny footprint is not missed coverage; the old advice read that
  footprint as "the capture only ever looked into one narrow wedge".

**What the first "after" run still got wrong, and the fix.** Told in words that metres off a
guessed scale are not evidence, the advisor still cited them: "all 17 frames sit in a 0.6 m²
standpoint" (11299294; 0.6 m² is the room's footprint on a guessed scale, not the camera's) and "the
floor is entirely missing" (7614541). The floor shares rest on the same guess, since a floor cell
counts as seen only within 5 m of a camera: on the 15 clips with a map, the guess put one room
95 m high and made the 23-second walk through two rooms a 1.1 m path, and five clips read 99-100%
of the floor never seen, two of them (39936361, 29814959) with the floor in view in every frame.
So until densify measures the scale, the advisor is no longer given the sizes or floor shares at
all (`capture_map_metrics` leaves them out and says why), and the shot list no longer asks for a
floor sweep on a guessed scale. Instruction did not stop the misreading; removing the number did.

**Two more rounds.** Re-asked on the clips that had misread, the advice found new ones, each fixed
and re-asked until it stopped:

- With the sizes gone, it reported the guess itself ("no real-world scale came out of this
  capture"): the note on what was left out now says the depth stage measures the scale later and a
  guess says nothing about the capture.
- It read a small `path_jump` (the largest step between two frames) as a camera that never moved;
  the guide now calls it a check for teleports, not a measure of distance travelled.
- Given "frames_kept 30, candidates_fps 6.0" it called a 15-second clip "about five seconds of
  footage"; sampling now gives the seconds of video spanned and the frames kept per second.
- It read "2 of 12 directions faced", drawn from the only 2 frames placed, as the whole capture's,
  and called 2,079 sparse points "very thin" (sound solves here held 731 to 17,159); the guide now
  says the coverage describes the placed frames and a point count alone is no finding.

On the last round the remaining findings checked out against the frames: 11299294, for instance, is
a pan of 8.3 seconds from one spot, and the advice said so from the seconds and the four bearings
faced, not from any metre.

## Stage 1 on all 16 clips (the pipeline as it runs)

`pipeline/agent.py --stage reconstruct`, COLMAP 4.3.0.dev0 (built from source, CPU only), SIFT and
the global mapper, 2 fps sampled from 6 fps candidates (sharpest of each three).

| Clip | What it is | Frames placed | Directions faced | Floor never seen | Gate |
|---|---|---|---|---|---|
| 3769951 | living room, short walk | 20/20 | 2 of 12 | 41% | pass |
| 8580864 | loft, pan from one spot | 33/33 | 2 of 12 | 26% | pass |
| 32144167 | period bedroom, two beds | 25/25 | 4 of 12 | 65% | pass |
| 7614541 | kitchen-dining, slow slide | 30/30 | 1 of 12 | 100%* | pass |
| 11299294 | living-dining room | 17/17 | 4 of 12 | 30% | pass |
| 38413578 | hotel bedroom, dolly back through a door | 16/16 | 1 of 12 | 100%* | pass |
| 39936361 | bedroom with dresser | 14/14 | 3 of 12 | 100%* | pass |
| 37674124 | living room, TV playing | 12/12 | 2 of 12 | 8.5% | pass |
| 29814959 | hotel bedroom | 9/9 | 4 of 12 | 100%* | pass |
| 34208854 | white bedroom, lateral slide | 2/12 | 2 of 12 | - | **stop** (2 of 12 placed) |
| 27975843 (new) | open-plan studio, slow slide | 10/14, no points | - | - | **stop** (no points) |
| 27975847 (new) | the same studio, another slide | 18/18 | 2 of 12 | 100%* | pass |
| 29681899 (new) | hotel suite through an arch | 9/9 | 1 of 12 | 31% | pass |
| 35023423 (new) | near-empty bedroom, ceiling-heavy | 14/14 | 2 of 12 | 15% | pass |
| 7749088 (new) | bedside detail past a curtain | 22/22 | 1 of 12 | 20% | pass |
| 35308496 (new) | 23 s walkthrough, living room and bedroom | 48/48 | 8 of 12 | 34% | pass |

\* Before densify the capture map's scale is a guess (the phone held 1.4 m up); "floor never seen
100%" on these clips is the guess failing, not the camera never looking down.

This was the pipeline as it stood when the audit began. Stock footage is gentler than a phone in a
hand: 14 of 16 pass, every pass places every frame, and the two failures are both camera motion
rather than texture: two short slides (6 and 7 seconds) in which the camera moves little against how
far away the room is. The studio slide (27975843) stopped with "Reconstruction produced an empty
cloud", and the advisor read its `path_jump` of 0.0 as "zero recorded motion".

With the fallback chain the bench below led to, both now solve on most fresh runs: the white-bedroom
slide (34208854) 12 of 12 frames on 4 of 4, the studio slide (27975843) all 14 frames on 3 of 4,
and the agent's stage 1 passes the studio slide (see "The two clips the global mapper cannot
solve").

## Stage 1: every option on the same frames

A bench script (kept outside the repo; every option it ran is given below as COLMAP flags) runs COLMAP 4.3.0.dev0 on the CPU with the pipeline's
own frame extraction (2 fps, sharpest of each three candidates at 6 fps), sequential matching and
one option changed at a time, so a difference is the option's. Counts are frames placed in the main
model; a "failed" solve made no model.

**A warning that shaped everything below.** COLMAP's incremental mapper draws its first image pair at
random (`Mapper.random_seed` defaults to -1). On the same database, the slow studio slide (27975843)
placed all 14 frames with 4,196 points on seeds 0 and 4 and 11 frames with no point on seeds 1-3. A
first pass of this bench ran each option once and credited the small-baseline angles with "12 of 12
on the white-bedroom slide"; five seeds gave 2 to 5, and that claim went into a commit before it was
caught and corrected. Every incremental comparison below is therefore over seeds 0-2, and the
pipeline now runs the incremental mapper with three seeds and keeps the best. The global mapper is
given its seed (0 here; the pipeline tries 0-2), and on the two clips it cannot solve it failed on
every seed from 0 to 6. The matching itself also varies slightly between fresh runs (the bench's
and the pipeline's databases for 27975843 differ by 7 inlier matches in 90,000), which is why the
two hardest clips are also measured over repeated fresh runs.

### Which mapper

The **global mapper** (the default) placed every frame on 14 of 16 clips and failed outright on the
other two. It stays the default.

**Incremental, COLMAP's angles against small-baseline angles, seeds 0-2:**

| Clip | frames | plain, seeds 0/1/2 | small-baseline, seeds 0/1/2 |
|---|---|---|---|
| 11299294 | 17 | 17 (2,269 pts), 17 (2,269 pts), 17 (2,269 pts) | 17 (2,612 pts), 17 (2,612 pts), 17 (2,754 pts) |
| 27975843 | 14 | 7 (2,307 pts), 9 (2,555 pts), 14 (4,081 pts) | 0, 14 (1,784 pts), 0 |
| 27975847 | 18 | 18 (5,287 pts), 18 (5,287 pts), 18 (5,287 pts) | 18 (6,231 pts), 18 (3,742 pts), 18 (3,742 pts) |
| 29681899 | 9 | 0, 0, 0 | 9 (3,948 pts), 9 (3,948 pts), 9 (3,948 pts) |
| 29814959 | 9 | 4 (230 pts), 9 (3,969 pts), 4 (230 pts) | 9 (5,736 pts), 9 (6,356 pts), 9 (5,799 pts) |
| 32144167 | 25 | 25 (13,065 pts), 3 (8,563 pts), 3 (8,563 pts) | 25 (18,357 pts), 25 (18,279 pts), 25 (17,689 pts) |
| 34208854 | 12 | 2 (483 pts), 2 (591 pts), 2 (591 pts) | 4 (482 pts), 4 (482 pts), 12 (4,266 pts) |
| 35023423 | 14 | 14 (4,190 pts), 14 (4,190 pts), 14 (4,190 pts) | 14 (4,174 pts), 14 (4,175 pts), 14 (4,176 pts) |
| 35308496 | 48 | 35 (14,532 pts), 35 (13,374 pts), 35 (13,193 pts) | 48 (23,877 pts), 48 (23,753 pts), 35 (17,044 pts) |
| 37674124 | 12 | 12 (2,262 pts), 12 (2,262 pts), 12 (2,270 pts) | 12 (2,524 pts), 12 (2,525 pts), 12 (2,521 pts) |
| 3769951 | 20 | 20 (4,167 pts), 20 (4,175 pts), 20 (4,169 pts) | 20 (4,318 pts), 20 (4,318 pts), 20 (4,318 pts) |
| 38413578 | 16 | 2 (1,778 pts), 2 (1,778 pts), 2 (1,778 pts) | 2 (2,271 pts), 2 (2,271 pts), 2 (2,271 pts) |
| 39936361 | 14 | 14 (17,430 pts), 14 (12,297 pts), 14 (12,297 pts) | 14 (15,962 pts), 14 (15,962 pts), 14 (15,962 pts) |
| 7614541 | 30 | 0, 0, 0 | 30 (10,492 pts), 30 (10,571 pts), 30 (10,572 pts) |
| 7749088 | 22 | 0, 0, 0 | 22 (933 pts), 22 (933 pts), 22 (933 pts) |
| 8580864 | 33 | 15 (5,165 pts), 9 (759 pts), 13 (5,363 pts) | 33 (5,176 pts), 33 (5,189 pts), 33 (5,189 pts) |

- seed for seed: small-baseline placed more frames in 22 runs, fewer in 2, the same in 24
- best of three seeds (what the pipeline keeps): more on 6 clips, fewer on 0, the same on 10

The small-baseline angles (first pair 4 degrees instead of 16, points kept down to 0.5 instead of
1.5; COLMAP's FAQ gives this advice for scenes filmed from far away) placed every frame on every
seed of three clips where COLMAP's angles placed none on any seed. Lowering only the first pair's
angle was not enough. They lost seed for seed on one clip (the slow studio slide). So the fallback
after a failed global solve tries the small-baseline angles first and COLMAP's own second, three
seeds each, and stops at the first run that places every frame with at least 500 points.

### The two clips the global mapper cannot solve

| | 27975843 (slow slide, open-plan studio) | 34208854 (sideways slide, white bedroom) |
|---|---|---|
| global, seeds 0-6 | failed on every seed | failed on every seed |
| view-graph calibration, then global | failed ("no image pairs to calibrate") | placed 12/12 (about 4,150 points, 0.49 px) on the bench's database, both seeds; failed on another fresh run's database (no pair dropped) |
| incremental, COLMAP's angles, seeds 0-2 | 7, 9, 14 | 2, 2, 2 |
| incremental, small-baseline, seeds 0-2 | 0, 14, 0 | 4, 4, 12 |
| 3x the frames, global / incremental | failed / 40 of 40 | failed / 11 of 36 |
| ALIKED, global / incremental | failed / 14 | failed / 2 |
| LoMa-B128, global | failed | failed |
| DEGENSAC, global | failed | failed |

Before this work both stopped at stage 1: 27975843 because a global solve with frames and no point
was accepted, so the incremental fallback never ran (the run reported "no parallax"); 34208854
because nothing tried could place it. Fresh end-to-end runs of `reconstruct.py`, extracting and
matching the frames anew each time:

| Fresh runs of `reconstruct.py` | 34208854 (sideways slide) | 27975843 (slow slide) |
|---|---|---|
| before this work | stopped (2 of 12 placed) | stopped (frames placed, no point) |
| first fallback chain (bf29618: COLMAP's angles only) | 12/12 on 2 of 5 runs; 2/12 on the others (in the agent, its retry then placed 7 of 12 and stage 1 stopped at 58%) | 14/14 on 5 of 5 runs, about 4,000 points (the agent's stage 1 passes) |
| final chain (90bfef0: small-baseline angles first) | 12/12 on 4 of 4 runs, 3,600-4,300 points (three by calibration and the global mapper, one by the small-baseline angles on its first seed) | 14/14 on 3 of 4 runs (one on 247 points, at which the search no longer stops); 7 of 14 on the fourth, where no seed of either angle setting did better |

Both clips sit on a knife edge: the same frames matched afresh give slightly different databases,
and on these two that decides the solve. The fallbacks raise the odds; they cannot make a 6-second
slide that barely moves a reliable capture.

### What did not earn its place

- **ALIKED** (learned features, about twice as slow to extract): never placed a larger share of
  frames than SIFT with the same mapper; fewer points on all 14 clips both solved, at a higher reprojection
  error (0.48-1.45 px against 0.38-0.84). Removed from the retries; `--features aliked` remains.
- **3x the frames** (the denser retry): the same share placed as the default on all 12 clips it ran
  on, more points on 10, and neither hard clip rescued. Kept only for a video whose frames show
  fast turns, where its reasoning (consecutive frames sharing too little) holds; no test clip was
  like that.
- **LoMa** (COLMAP 4.2's learned matcher, ECCV 2026): LOMA_B ran this 15 GB box out of memory;
  LOMA_B128 took about 9 s a frame single-threaded and placed nothing the default did not.
- **View-graph calibration first**: the same frames on 4 clips and fewer points on all 4 (one solve,
  29814959, kept a single point, on a repeat too). As a fallback after the global mapper fails it
  is the only thing that solved the white-bedroom slide with the global mapper, so it is that.
- **AnyCalib single-view calibration**: the same frames, fewer points on 2 of 3.
- **DEGENSAC** (COLMAP 4.2's two-view estimator for plane-dominated scenes): on 6 clips, the same
  frames on the 4 that solve, points within 1%, neither failure rescued.
- **Sharpest of three** against uniform sampling: the same frames on 3 clips, more points on 2. It
  costs nothing, so it stays.

## What the field says (October 2026), layer by layer

Sources were read in full where Firecrawl could fetch them (GitHub issues and READMEs, arXiv,
COLMAP's changelog and docs, Anthropic's docs, Hacker News); Reddit, X and LinkedIn pages cannot
be fetched, so their threads are quoted from search excerpts only and weighed accordingly.

### Capture (the phone)

| Question | What the evidence says | Source |
|---|---|---|
| Can a non-LiDAR iPhone use RoomPlan or ARKit scene depth instead of our own models? | No: RoomPlan and ARKit scene reconstruction require the LiDAR scanner (Pro models only). The project's reason for running MoGe-2-small on the phone holds. | KIRI Engine and 3D Mag guides citing Apple's docs |
| Android equivalent | ARCore's Depth API works without a depth sensor but gives a 160x90 depth image, most accurate at 0.5-5 m. Same conclusion. | developers.google.com/ar/develop/depth; maziminds |
| Exposure and white balance | Practitioners lock both for interiors: "Locking exposure, white balance, focus was a huge win for us"; "the white balance seems to be different on some picture"; a working FOSS pipeline on HN captured "with exposure locked, framerate locked, focus locked". ARKit allows it since iOS 16 through `configurableCaptureDeviceForPrimaryCamera`. The app did not lock them. **Changed: the app now holds exposure and white balance while recording.** | r/GaussianSplatting excerpts; HN item 48396792; WWDC22 "Discover ARKit 6" |
| Detector on the phone | YOLOE-26 leads YOLOE-11 "at every matching scale" on LVIS (26m 35.4 mAP, 26s 30.8; GDINO-T 27.4 on the same split at 155-232 M parameters). A drop-in upgrade of the phone's YOLOE-11m, once converted to Core ML and checked on our rooms. | docs.ultralytics.com/models/yoloe (YOLO26 paper) |
| One model for detector + refiner | EfficientSAM3 (TinyViT + MobileCLIP-S0) and SAM3-LiteText distil SAM 3's text-prompted segmentation for devices; SAM 3 itself is 848 M parameters and 23-40 s per tracked frame on a CPU (sam3.cpp). Not ready to replace YOLOE + MobileSAM on a phone today. | github.com/SimonZeng7108/efficientsam3; github.com/PABannier/sam3.cpp |

### Stage 1 (cameras)

| Question | What the evidence says | Source |
|---|---|---|
| What changed in COLMAP since the survey | 4.1: SIFT + LightGlue fixed. 4.2: the global mapper returns every connected component; LoMa learned features; 6-point solvers "improving reconstruction when camera intrinsics are unknown"; optional DEGENSAC for plane-dominated scenes. 4.2.1: global-mapper bundle-adjustment regression fixed. AnyCalib single-view calibration. | github.com/colmap/colmap/releases; colmap.github.io/changelog.html |
| Should `view_graph_calibrator` run before the global mapper? | COLMAP's FAQ recommends it when intrinsics are unknown (ours always are for stock video), and the global mapper's own log says so when it fails. **Measured here: as a first step it cost points on every clip that solved without it; as a fallback after the global mapper fails it solved the white-bedroom slide on some fresh runs.** It is now that fallback. | colmap.github.io/faq.html; colmap#4557 |
| Small-baseline triangulation angles | COLMAP's FAQ: for images "taken from far distance with respect to the scene, you can try to reduce the minimum triangulation angle"; a video-tracking paper (SuperPoint-E) sets the first pair's angle to 8 degrees, a splatting tutorial to 2. **Measured here over three seeds: more frames than COLMAP's angles in 22 runs, fewer in 2.** | colmap.readthedocs.io FAQ; arXiv 2602.04108; smartdatascan.com |
| Global or incremental mapper on video? | Mixed in the field: GLOMAP is "way faster ... but has the odd fluke"; on one 2-fps video "with glomap one gets the crazy outliers", on another GLOMAP placed 98/98 against COLMAP's 83/98. **Measured here: global placed every frame on 14 of 16 clips; incremental with COLMAP's angles lost frames on 7 of them, and its result depends on the seed.** | colmap/glomap#128; r/GaussianSplatting excerpt |
| LoMa (ECCV 2026 oral, MIT) | New state of the art in matching; in COLMAP as LOMA_B (a 1.29 GB DeDoDe-G descriptor that ran this 15 GB box out of memory) and LOMA_B128. **Measured here: about 9 s a frame on this CPU, and it solved neither hard clip.** Worth a GPU trial, not a CPU default. | github.com/davnords/LoMa; arXiv 2604.04931 |
| Replace COLMAP with a feed-forward model? | Depth Anything 3 beats VGGT by 35.7% on pose accuracy; long-video variants (DA3-Streaming, VGGT-Long, pi3-Long, LoGeR, VGGT-Omega) appeared in 2026. Licences matter: DA3-Large is CC BY-NC 4.0 (non-commercial); DA3-Small and DA3-Metric-Large are Apache-2.0. COLMAP's reprojection error, track lengths and units story are what every gate here reads; a feed-forward solve has none of them. Keep COLMAP; MapAnything stays the gap-filler. | arXiv 2511.10647; github.com/bytedance-seed/depth-anything-3; SpatialBench arXiv 2605.27367 |
| Fixed-rate sampling | "an adaptive, coverage-aware sampling strategy would be better" (HN); tools exist (adaptive-frame-extractor, AliceVision keyframes). Our sharpest-of-three is a weaker form of it. | HN 48396792 |

### Stage 2 (depth and labels)

| Question | What the evidence says | Source |
|---|---|---|
| MoGe-2 against the alternatives | Per frame it is near the top indoors (NYUv2 delta1 0.967; UniDepthV2-L 0.989; Depth Pro 0.953), but the least consistent across video frames of those compared (ScanNet temporal error 2.56 against 1.41 and 1.09). The pipeline's per-keyframe scale agreement (7% on the T4 run) is the measure that catches this. | Video Depth Anything README table |
| Metric scale without familiar objects | MoGe's author: the model "cannot reliably infer scale" for a scene lacking "familiar references" -> an empty room is the hardest case for metric depth; DA3's metric model is also off by over 100% under 0.5 m. | microsoft/MoGe#75; DA3#244 |
| SAM 3 as the awaited replacement for GroundingDINO + SAM 2.1 + tracking | One model: text prompt -> every instance's mask and ID through a video; users preferred it 3:1 over OWLv2; SAM 3.1 (Mar 2026) tracks many objects jointly and faster. Weights are gated on Hugging Face (blocked in this cloud box), under Meta's SAM licence. Worth its own GPU trial on Colab against the current three-model chain. | github.com/facebookresearch/sam3; ai.meta.com/blog/segment-anything-model-3 |
| CLIP for the crop vote | SigLIP 2 is ahead of CLIP L/14 on zero-shot ImageNet (78.1 vs 74.5): a modest gain for a vote that has flipped nothing in 64 tries. Not worth a change now. | arXiv 2502.14786 |

### Stage 3 (structure)

| Question | What the evidence says | Source |
|---|---|---|
| Learned layout | SpatialLM 1.1 on point clouds from monocular video, zero-shot: walls F1 68.9, doors 49, windows 47-51; beds 96.8 but wardrobes 29-40 and cabinets 11-15. Measured walls with our own rollback score are not beaten by it on furniture; it could add doors and windows. | github.com/manycore-research/SpatialLM |
| Single-image 3D for the clean models | SAM 3D Objects (CVPR 2026 best-paper honourable mention) makes a textured mesh of a masked object from one image; 5:1 human preference. A candidate for the "clean model" each piece can swap to. | github.com/facebookresearch/sam-3d-objects |

### Stage 4 (splat)

| Question | What the evidence says | Source |
|---|---|---|
| OpenSplat | AGPLv3; no depth supervision, no exposure compensation, no MCMC. | github.com/WebODM/OpenSplat |
| gsplat | Apache-2.0; MCMC (1 M gaussians matched 3.2 M default ones at a third of the memory); bilateral-grid exposure compensation; depth loss; CUDA only. | docs.gsplat.studio |
| Brush | Apache-2.0; WebGPU, so Mac, Windows, Linux, Android and the browser; COLMAP input with masks; "training ... generally faster than gsplat"; a practitioner found it "very straightforward ... gave good results". | github.com/ArthurBrussee/brush; HN 48396792 |
| LaMa for filling unseen floor and wall | Big-LaMa "produces repetitive artifacts" on large masks; practitioners' open choice for inpainting in 2026 is FLUX.1 Fill. Our fill review rejected all 3 LaMa fills on the T4 run, so the step currently ships nothing; the mixed scene's plain-paint fallback is what users see. | arXiv 2210.17106; r/StableDiffusion "Best model for inpainting in 2026" (excerpt) |
| Depth priors for indoor splats | DN-Splatter: adding depth supervision raised PSNR (24.77 vs 24.44) and cut depth error by two thirds on indoor scenes. Our MoGe depth is already computed per frame and unused in training. | arXiv 2403.17822 |

### The judge

| Question | What the evidence says | Source |
|---|---|---|
| How reliable are vision-language judges? | The best of 20 reached 63.1% (chance 32.2%) on criterion-conditioned comparisons, with positional bias towards some answer letters. Matches this project's own 45.8% measure of judges against true geometry. Our splat choice already votes over swapped orders; the start view does not. | VisionQ, arXiv 2610.00666 |
| How to prompt them | Anthropic: "Providing context or motivation behind your instructions, such as explaining to Claude why such behavior is important, can help Claude better understand your goals"; "tell Claude what to do instead of what not to do". -> every prompt now says where it sits and what its answer changes. | platform.claude.com prompting best practices |
| Image size | Claude 4.7 and later read up to 2576 px on the long side; the relay shrank images to 1600. **Changed to 2576.** | platform.claude.com vision docs |

## Decisions this puts in front of the owner

These need a GPU, gated weights, a licence call or a phone, so they are proposals with the evidence
behind them, not changes made here.

1. **The splat trainer: OpenSplat (AGPLv3) or gsplat / Brush (Apache-2.0).** OpenSplat has no depth
   supervision, no exposure compensation and no MCMC; gsplat has all three (MCMC matched 3.2 M default
   gaussians with 1 M at a third of the memory; DN-Splatter's depth loss raised indoor PSNR and cut
   depth error by two thirds), and the MoGe depth it would use is already computed per frame. Brush
   trains on any GPU including a Mac's and reportedly faster than gsplat. AGPLv3 matters if the
   pipeline ever runs as a service. Proposed test: the walkthrough and the living room on the Colab T4,
   gsplat (MCMC + bilateral grid + MoGe depth loss) against today's quick OpenSplat run, judged by the
   existing unseen-view SSIM and Claude's swapped-order vote.
2. **SAM 3 (or SAM 3.1) in place of GroundingDINO + SAM 2.1 + video tracking.** One model, text prompt
   to every instance's mask and identity through the video; users preferred it 3:1 over OWLv2. The
   weights are gated on Hugging Face under Meta's SAM licence (needs an access request, and the cloud
   box cannot reach Hugging Face at all). Proposed test on Colab: the same object list through both
   chains on the walkthrough, counting boxes the structure review keeps.
3. **Depth: stay on MoGe-2.** Depth Anything 3's best models are non-commercial (CC BY-NC 4.0); its
   metric model is Apache-2.0 but is no better at close range (over 100% off under 0.5 m). MoGe-2 is
   accurate per frame (NYUv2 delta1 0.967) and the pipeline already measures its weakness (scale
   agreement across keyframes). Revisit if the scale agreement gate starts stopping real captures.
4. **The phone's detector: YOLOE-11m to YOLOE-26m.** Same Ultralytics family and Core ML conversion
   path, better at every size on LVIS. Needs a conversion and a check on our rooms on a phone.
5. **The model behind the advisor.** advisor.py pins one model (`MODEL`); newer ones have shipped
   since. Changing it is a one-line edit, best made with a before/after probe like the one in this
   report (each question asked against the stage-1 spaces of all 16 clips, with the old and the new).
6. **The capture app's exposure lock** was written here without a Swift toolchain: it follows Apple's
   documented ARKit 6 pattern, but it must be built in Xcode and checked on a phone before release (the
   recorded per-frame `exposure` in capture.json should be constant within a recording).
7. **A live "the view is not shifting" rule in the capture app.** Both clips the pipeline still
   struggles with are slides of 6-7 seconds where the camera moves little against how far away
   the room is; the final fallback chain solves them on most fresh runs (4 of 4 and 3 of 4), but
   only just, and a person's capture should not depend on that. The app already catches the
   neighbours of this failure: turning on the spot (`spinning`), recordings under 20 s (`tooShort`,
   at the end), and coverage. It does not
   catch a slow glide live. A rule comparing the distance moved over the last few seconds with the
   typical depth in view (ARKit gives both) would, but its threshold needs tuning on real app
   captures, and the Swift package cannot be built here, so it is a proposal, not a change.
8. **Masks for people and pets in splat training.** The naming prompt now names them as "loose", so
   densify knows their pixels; OpenSplat, gsplat and Brush all accept per-frame masks. Writing those
   masks would keep a person who walked through from becoming a ghost in the splat.

## What changed in the code

On branch `claude/eager-goldberg-9ixhyg`, in order:

| Commit | Change |
|---|---|
| d1c2b7b | The capture app holds exposure and white balance while recording (ARKit 6 `configurableCaptureDeviceForPrimaryCamera`); written without a Swift toolchain. |
| 01cfe2f | Every Claude question reworded for any capture, with its place and consequence; frame choice spread first to last (it never showed the last 12 frames of a 48-frame capture); empty rooms keep an empty furniture list; a zero-point solve stops with a parallax message; relay images at 2576 px. |
| 7348911 | A global solve with frames and no point falls back to the incremental mapper; the advisor no longer sees sizes or floor shares measured on a guessed scale; the shot list skips its floor sweep then; a run that dies before COLMAP makes a model reports no "0% of 0 frames". |
| 583c087 | The advice is given the seconds of video spanned, not the candidates' frame rate; path_jump defined as a teleport check; a guessed scale is not reported as a finding. |
| 8f7a9a3 | The coverage describes the placed frames; a sparse point count alone is no finding. |
| 1f610e3 | `--mapper small-baseline`; the ALIKED retry removed; reconstruct.py records why it rejected the global solve, and the advice tells "no point" from "joined pieces". (Its claim about the white-bedroom slide was a single random draw; corrected in the next commit.) |
| bf29618 | View-graph calibration as a fallback after the global mapper fails; the incremental mapper run with seeds 0-2, keeping the best; a model under 100 points is no solve. |
| 90bfef0 | After a failed global solve, the small-baseline angles first, then COLMAP's own; README and a new test suite (18 suites, all passing). |
| 2b7a55a | Code comments cite the full seeded bench. |
| (with this report) | The seeded search stops early only at a model with every frame and at least 500 points; this report. |

## Appendix: the single-run bench

Every option once, on the same frames. The global columns use seed 0; the incremental columns are single
draws of a random solve, so read them with the seeded table above. "N pieces" counts the models with
frames and points beyond the main one.

| Clip | SIFT, global (default) | SIFT, incremental | SIFT, incremental, small-baseline angles | ALIKED, global | ALIKED, incremental | 3x frames, global | 3x frames, incremental | LoMa-B128, global | DEGENSAC | view-graph calibration | AnyCalib | no sharpness pick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11299294 | 17/17, 2,036 pts, 0.62 px | 17/17, 2,269 pts, 0.62 px | 17/17, 2,612 pts, 0.59 px | 17/17, 1,479 pts, 1.05 px | 17/17, 2 pieces, 1,694 pts, 0.90 px | 50/50, 4,662 pts, 0.67 px |  |  |  |  |  |  |
| 27975843 | failed (0/14) | 14/14, 2 pieces, 4,081 pts, 0.56 px | 13/14, 0 pts, 0.00 px | failed (0/14) | 14/14, 1,666 pts, 0.94 px | failed (0/40) | 40/40, 2 pieces, 5,255 pts, 0.63 px | failed (0/14) | failed (0/14) |  |  |  |
| 27975847 | 18/18, 2,516 pts, 0.47 px | 18/18, 5,287 pts, 0.60 px | 18/18, 2 pieces, 3,742 pts, 0.50 px | 18/18, 1,587 pts, 0.89 px | 18/18, 2 pieces, 1,679 pts, 0.93 px | 52/52, 2,786 pts, 0.53 px |  |  |  |  |  |  |
| 29681899 | 9/9, 3,103 pts, 0.55 px | failed (0/9) | 9/9, 2 pieces, 3,948 pts, 0.54 px | 9/9, 969 pts, 1.22 px | failed (0/9) | 25/25, 6,005 pts, 0.71 px |  |  |  |  |  |  |
| 29814959 | 9/9, 5,717 pts, 0.38 px | 4/9, 2 pieces, 230 pts, 0.39 px | 9/9, 5,799 pts, 0.38 px | 9/9, 2,391 pts, 0.48 px | 9/9, 2,459 pts, 0.47 px | 26/26, 17,598 pts, 0.41 px |  |  | 9/9, 5,729 pts, 0.37 px | 9/9, 1 pts, 0.05 px | 9/9, 819 pts, 0.35 px | 9/9, 6,106 pts, 0.35 px |
| 32144167 | 25/25, 4,721 pts, 0.71 px | 25/25, 2 pieces, 13,147 pts, 0.38 px | 25/25, 2 pieces, 18,415 pts, 0.36 px | 25/25, 3,148 pts, 0.72 px | 25/25, 2 pieces, 2,702 pts, 0.80 px | 74/74, 22,623 pts, 0.74 px |  |  |  |  |  |  |
| 34208854 | failed (0/12) | 2/12, 483 pts, 0.33 px | 12/12, 3,077 pts, 0.66 px | failed (0/12) | 2/12, 243 pts, 0.42 px | failed (0/36) | 11/36, 2 pieces, 1,025 pts, 0.52 px | failed (0/12) | failed (0/12) |  |  |  |
| 35023423 | 14/14, 3,970 pts, 0.69 px | 14/14, 4,167 pts, 0.68 px | 14/14, 4,174 pts, 0.68 px | 14/14, 1,954 pts, 1.45 px | 14/14, 2,571 pts, 1.08 px | 41/41, 10,796 pts, 0.76 px |  |  | 14/14, 3,973 pts, 0.68 px |  |  |  |
| 35308496 | 48/48, 16,018 pts, 0.72 px | 30/48, 3 pieces, 11,426 pts, 0.88 px | 35/48, 3 pieces, 16,398 pts, 0.83 px | 46/48, 5,641 pts, 1.21 px | 24/48, 3 pieces, 4,567 pts, 1.12 px |  |  |  |  |  |  |  |
| 37674124 | 12/12, 2,242 pts, 0.84 px | 12/12, 2 pieces, 2,262 pts, 0.86 px | 12/12, 2,532 pts, 0.85 px | 12/12, 1,538 pts, 1.11 px | 12/12, 2 pieces, 1,886 pts, 1.13 px | 36/36, 7,654 pts, 0.88 px | 36/36, 2 pieces, 7,572 pts, 0.89 px | 12/12, 2,103 pts, 1.07 px |  | 12/12, 1,920 pts, 0.70 px | 12/12, 2,276 pts, 0.87 px | 12/12, 2,314 pts, 0.82 px |
| 3769951 | 20/20, 4,211 pts, 0.40 px | 20/20, 4,169 pts, 0.39 px | 20/20, 4,318 pts, 0.40 px | 20/20, 2,466 pts, 0.57 px | 20/20, 2,574 pts, 0.57 px | 60/60, 11,783 pts, 0.44 px |  | 20/20, 3,476 pts, 0.56 px | 20/20, 4,210 pts, 0.40 px | 20/20, 4,043 pts, 0.33 px | 20/20, 4,207 pts, 0.39 px | 20/20, 4,072 pts, 0.39 px |
| 38413578 | 16/16, 3,893 pts, 0.52 px | 2/16, 1,778 pts, 0.33 px | 2/16, 2,271 pts, 0.25 px | 16/16, 1,473 pts, 1.01 px | failed (0/16) | 47/47, 8,718 pts, 0.56 px |  |  |  |  |  |  |
| 39936361 | 14/14, 17,159 pts, 0.56 px | 14/14, 2 pieces, 17,430 pts, 0.48 px | 14/14, 2 pieces, 15,962 pts, 0.47 px | 14/14, 2,251 pts, 0.75 px | 14/14, 1,817 pts, 0.65 px |  |  |  |  |  |  |  |
| 7614541 | 30/30, 6,541 pts, 0.57 px | failed (0/30) | 30/30, 10,494 pts, 0.50 px | 30/30, 1,877 pts, 0.94 px | failed (0/30) |  |  |  |  |  |  |  |
| 7749088 | 22/22, 731 pts, 0.80 px | failed (0/22) | 22/22, 933 pts, 0.78 px | 22/22, 338 pts, 1.07 px | failed (0/22) | 64/64, 928 pts, 0.86 px |  |  |  |  |  |  |
| 8580864 | 33/33, 10,351 pts, 0.54 px | 11/33, 3 pieces, 2,493 pts, 0.35 px | 33/33, 5,176 pts, 0.46 px | 33/33, 2,347 pts, 0.70 px | 15/33, 3 pieces, 1,480 pts, 0.52 px |  |  |  | 33/33, 10,315 pts, 0.54 px | 33/33, 1,301 pts, 0.41 px |  |  |

- SIFT, incremental against the default on 16 clips: placed a larger share of its frames on 2, a smaller on 7, the same on 7 (of those, more points on 6, fewer on 1)
- SIFT, incremental, small-baseline angles against the default on 16 clips: placed a larger share of its frames on 2, a smaller on 2, the same on 12 (of those, more points on 10, fewer on 2)
- ALIKED, global against the default on 16 clips: placed a larger share of its frames on 0, a smaller on 1, the same on 15 (of those, more points on 0, fewer on 13)
- ALIKED, incremental against the default on 16 clips: placed a larger share of its frames on 2, a smaller on 6, the same on 8 (of those, more points on 0, fewer on 8)
- 3x frames, global against the default on 12 clips: placed a larger share of its frames on 0, a smaller on 0, the same on 12 (of those, more points on 10, fewer on 0)
- 3x frames, incremental against the default on 3 clips: placed a larger share of its frames on 2, a smaller on 0, the same on 1 (of those, more points on 1, fewer on 0)
- LoMa-B128, global against the default on 4 clips: placed a larger share of its frames on 0, a smaller on 0, the same on 4 (of those, more points on 0, fewer on 2)
- DEGENSAC against the default on 6 clips: placed a larger share of its frames on 0, a smaller on 0, the same on 6 (of those, more points on 2, fewer on 2)
- view-graph calibration against the default on 4 clips: placed a larger share of its frames on 0, a smaller on 0, the same on 4 (of those, more points on 0, fewer on 4)
- AnyCalib against the default on 3 clips: placed a larger share of its frames on 0, a smaller on 0, the same on 3 (of those, more points on 1, fewer on 2)
- no sharpness pick against the default on 3 clips: placed a larger share of its frames on 0, a smaller on 0, the same on 3 (of those, more points on 2, fewer on 1)
