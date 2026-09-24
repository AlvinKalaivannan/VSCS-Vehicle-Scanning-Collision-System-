# VSCS capture checklists

**Task P1-T1.** Acceptance criterion: *the developer has reviewed this.* Read it once at
a desk, then again on site. Print it or keep it open on a second device — not on the
phone that is recording.

Capture days are the one irreversible part of this project. Reconstruction, segmentation
and every risk number are downstream of footage that can only be re-shot by getting the
van out again, and §6 puts all real-world capture in **October, before snow**. A wasted
capture day costs weeks.

> **Two things are not yet decided and block a real capture day.** See
> [Open questions](#open-questions) at the bottom. Please settle them before booking a day.

---

## 0. Safety rules that override everything

These come from CLAUDE.md §9 and are not negotiable by convenience on the day.

- [ ] **A licensed driver operates the van. The phone operator is never the driver.** Two
      people minimum for any lot run.
- [ ] **Walking speed only, under 5 km/h.** If the engine note rises, you are going too
      fast.
- [ ] **Empty lot, with permission to use it.** Confirm permission before the day, not on
      arrival.
- [ ] **No people behind or beside the moving vehicle. Ever.** The spotter stands
      *forward* of the front axle and off to the side, in the driver's mirror, never in
      the path.
- [ ] **Obstacles are soft and cheap only** — traffic cones, cardboard boxes, a foam or
      PVC pole. Nothing that damages the van if touched, nothing that damages a person.
- [ ] **The phone is never handled while the vehicle is moving.** Mount it, start
      recording, then drive. Stop the vehicle before touching the phone.
- [ ] **Check current Ontario distracted-driving rules** before any in-vehicle screen use.
      A mounted phone recording is not the same as a phone being operated; know the
      difference before you are asked.
- [ ] **Stop immediately** on rain, ice, poor visibility, or if the lot stops being empty.

**Dynamic-object clips (P4-T4) are not shot on these days.** A person near a moving
vehicle is exactly what §9 forbids. Those clips use a person walking at a safe distance
with the **vehicle stationary**, or public footage, or CARLA.

---

## 1. Before either capture day

### Equipment

- [ ] Phone, fully charged, **plus a power bank**. 4K video eats battery and a dead phone
      mid-session ends the day.
- [ ] At least 64 GB free storage. Check *actual* free space, not the plan.
- [ ] Secure phone mount for the van (suction or vent), tested on a short drive
      beforehand. A mount that slips mid-pass invalidates the run's extrinsics.
- [ ] **Tape measure** (5 m minimum), ideally a second one.
- [ ] **Printed ArUco markers** — see below.
- [ ] Rigid flat backing for each marker (clipboard, foam board, stiff card). Paper that
      curls is a curved marker, and a curved marker is a wrong scale.
- [ ] Chalk or low-tack tape to mark obstacle positions on the ground.
- [ ] Traffic cones, cardboard boxes, foam or PVC pole (lot day).
- [ ] Checkerboard for calibration, on rigid backing.
- [ ] This checklist, on paper or a second device.

### Markers (this is what makes the model metric — R-02)

- [ ] Generate sheets: `python scripts/make_markers.py --out data/raw/markers`
- [ ] Print at **100% scale / no fit-to-page**. "Shrink to fit" silently rescales your
      marker and therefore your entire vehicle model.
- [ ] **Measure the printed marker's black square with a ruler or calipers** and write the
      measured value into `configs/recon.yaml` under `scale.marker.side_length_m`. Do not
      trust the nominal size. Each sheet prints its intended size on it so you can check.
- [ ] Mount each on rigid backing. Confirm flat.
- [ ] **At least 3 markers**, different IDs, spread around the vehicle — not clustered on
      one side. §8.2 steps down to tape-measured dimensions if marker detection fails in
      over half the frames, and three spread markers is what avoids that.

### Camera calibration (P0-T7) — do this first, and never again after

- [ ] Lock focus and exposure. Main lens only. **No zoom, ever** (R-04).
- [ ] Shoot 15+ stills of the checkerboard: tilted, rotated, and pushed into all four
      corners of the frame. Fronto-parallel shots alone leave the focal length poorly
      constrained.
- [ ] Run `python scripts/calibrate.py --images <folder> --device <name>`. It **refuses**
      to write unless the mean reprojection error is under 0.5 px.
- [ ] **Use identical camera settings for every capture afterwards.** A calibration
      belongs to one lens at one resolution with one focus setting. Change any of those
      and the calibration is void and must be redone.

### Phone settings (lock these, then do not touch them)

| Setting | Value | Why |
|---|---|---|
| Lens | Main / 1× only | Ultrawide and tele are different cameras with different intrinsics (R-04) |
| Zoom | None | Digital zoom crops and changes effective intrinsics |
| Focus | **Locked** (AF-L) | Autofocus changes focal length mid-clip and voids the calibration |
| Exposure | **Locked** (AE-L) | Auto-exposure changes brightness between frames and hurts feature matching |
| Stabilisation | **Off** if possible | EIS warps the image per-frame; the calibration no longer describes it |
| Resolution / fps | Scan: 4K30. Lot: 1080p60 | Scan needs detail; lot runs need timing and less motion blur |
| HDR | Off | Per-frame tone changes confuse matching |
| Grid / overlays | Off | Nothing burned into the pixels |

Stabilisation is the one to check carefully. On many phones it cannot be disabled, in
which case **record it in the devlog** — it becomes a known limitation, and if
reconstruction quality is poor it is the first suspect.

---

## 2. Scan day — the van (P1-T2)

**Acceptance:** raw files ingested, checksummed in `data/MANIFEST.md`, backed up to two
locations. **Target:** ≥90% of frames registered, and all four measured dimensions within
±2 cm (P1-T5, P1-T6).

### Conditions

- [ ] **Overcast**, or open shade. Specular highlights move with the camera, which is
      precisely what breaks feature matching (R-01).

> **This van is dark maroon and glossy, so read R-01 this way.** Dark paint will not blow
> out to white the way a light van does — that failure is not your problem. Your problem
> is the opposite: glossy dark panels behave like **mirrors**. They reflect the sky,
> trees and buildings, those reflections slide across the panel as you walk, and SfM
> tries to match them as if they were surface features. Under flat overcast the same
> panels instead go **uniformly dark with almost no texture at all**.
>
> Either way the large flat panels are the weak surface, so the things that actually help
> are **overlap, ground texture and markers** — not merely avoiding sun:
>
> - [ ] Walk **slower** and take **more overlapping views** than feels necessary. Overlap
>       is the single best defence when the surface itself is uncooperative.
> - [ ] Put **extra markers and textured objects on the ground** all round the van. If the
>       panels give SfM nothing, the ground must.
> - [ ] Expect holes in the middle of big flat panels. That is R-01 behaving as predicted,
>       not a mistake you made. The bumpers, wheels, mirrors and trim — the parts with
>       actual geometry, and the parts VSCS cares about — will reconstruct far better than
>       the door skins.
> - [ ] *Optional, your call:* low-tack painter's tape crosses on the largest blank panels
>       give SfM real features and peel off afterwards. It is a standard photogrammetry
>       trick and it does not change the geometry, since the tape is flat on the surface.
>       **Only if you are happy putting tape on the van** — test one piece somewhere
>       inconspicuous first, and remove it all the same day.
- [ ] Van **clean and dry**. Water droplets and dirt are features that move.
- [ ] Park well away from walls, with room to walk a full loop plus a couple of metres.
- [ ] Ground with visible texture (asphalt is ideal). A smooth clean concrete floor gives
      SfM nothing to hold on to — this is why markers go on the ground too.

### Measure the van first, by hand

Do this **before** recording, and write the numbers straight into today's devlog. These
are the ground truth that P1-T6's ±2 cm gate is checked against, and they take five
minutes.

- [ ] **Length** (front bumper to rear bumper, along the vehicle) — record cm
- [ ] **Width** (excluding mirrors; note mirror-to-mirror separately) — record cm
- [ ] **Height** (ground to roof, highest point, note if there is a roof rack) — record cm
- [ ] **Wheelbase** (front axle centre to rear axle centre) — record cm
- [ ] Measure each **twice**. If the two readings differ by more than 1 cm, measure again.
      You cannot validate a model to ±2 cm with ground truth that is only good to ±3 cm.

Also note, because the vehicle frame origin depends on it (§4.1: origin on the ground
below the **centre of the rear axle**):

- [ ] Rear overhang: rear axle centre to rear bumper — record cm
- [ ] Front overhang: front axle centre to front bumper — record cm

### Marker placement

- [ ] 3+ markers flat **on the ground** around the van, spread out, each visible from a
      good part of at least one loop.
- [ ] Note which marker ID sits where, roughly, in the devlog.
- [ ] Optionally tape one marker to a flat body panel — helpful, but ground markers are
      the ones that also pin down the ground plane.

### Recording — doors closed

Walk **slowly**. Slow is the single highest-value thing you control: it reduces motion
blur and rolling-shutter skew at once (R-03).

- [ ] **Loop 1 — low**, camera at about knee height, angled slightly up. Catches sills,
      wheels, underbody edge, bumper undersides.
- [ ] **Loop 2 — mid**, camera at chest height, level. The main body pass.
- [ ] **Loop 3 — high**, camera above head, angled down. Roof, bonnet, mirror tops.
- [ ] 2–3 complete loops in total, each a **continuous take** — do not stop and restart
      mid-loop.
- [ ] Keep the van filling most of the frame. Overlap generously between consecutive
      views; an SfM pipeline would rather have too many frames than a gap.
- [ ] **Slow down and take extra passes at the parts that matter most**: both rear bumper
      corners, both mirrors, the four wheels. These are the components the whole project
      is about, and thin or protruding parts are the hardest to reconstruct.
- [ ] Windows and glass will reconstruct badly. Expected (R-01), not a failure. Do not
      waste the day fighting it.

### Recording — doors and mirrors open

Needed for the joints at P2-T7, and — for the sliding door — for the **second collision
model** (ADR 0004).

> **The sliding-door-open pass is load-bearing, not a quick extra.** The door runs on a
> curved track, so it is not modelled as a joint. Instead there are two separately scanned
> collision models, closed and open, and the one used per run is whichever you wrote down.
> That means the door-open pass has to be **good enough to reconstruct on its own** — same
> care, same overlap, same three heights as the main loops. A sloppy pass here means no
> door-open model at all.

- [ ] Sliding door fully open: a **full-quality** slow pass along that side, not a sweep.
- [ ] **Measure the door's outward throw** with a tape: body skin to door skin at full
      open, versus closed. One minute, and it is the measurement that justifies ADR 0004.
      Write it into the devlog and into `configs/model.yaml`
      (`variants.sliding_door_right.measured_outward_throw_m`).
- [ ] Rear doors fully open: one slow pass at the rear.
- [ ] Mirrors folded: one short pass of each.
- [ ] **Record, for each, which parts are open, in the devlog.** Joint states are entered
      by hand per run in v0.1 (R-15); if nobody wrote down what was open, the clip is
      geometry with no known configuration.

### Before leaving

- [ ] Play back one clip on the phone. Check it is in focus, correctly exposed, and not a
      0-byte file.
- [ ] Confirm file count and total size look right.
- [ ] Photograph the whole scene, including where the markers were. Cheap, and settles
      arguments later.

---

## 3. Lot day — driving runs (P1-T3)

**Acceptance:** ≥10 passes recorded, ground-truth positions in
`data/raw/lot_*/gt.yaml`, backed up. Cone positions must later land within ±25 cm at
≤3 m (P3-T2).

### Setup

- [ ] Empty lot, permission confirmed, no through traffic.
- [ ] Phone mounted **rear-facing** — this is a reversing system. Mount rigidly, and once
      mounted **do not move it between passes within a session**: moving it changes the
      camera-to-vehicle transform, which invalidates every extrinsic estimate for the
      session.
- [ ] Note the mount position: roughly how high, how far back, and its rough angle. Even
      an approximate note helps sanity-check ego-motion later.
- [ ] Phone settings locked as above (1080p60 for lot runs).
- [ ] IMU logging running — **see the open question about this below.**

### Obstacles, at measured positions

The whole point of the lot day is that the answers are known. Positions go in
`data/raw/lot_<date>/gt.yaml`.

- [ ] Choose a fixed origin on the ground and mark it (chalk X). Every measurement is
      from there. Write down what and where it is.
- [ ] Place each obstacle, then **tape-measure it to the origin in two axes** and record
      it. Not paced out, not estimated.
- [ ] Include all three difficult categories, because R-07 is about exactly these:
  - [ ] **Thin pole** (foam or PVC) — the case a bounding box misses entirely
  - [ ] **Low kerb** or a plank on edge — the underbody case (P4-T6)
  - [ ] **Boxes / cones** — the easy baseline case
- [ ] Mark each obstacle's footprint on the ground with chalk or tape, so a nudged cone is
      visible and can be re-measured rather than silently corrupting the ground truth.
- [ ] Photograph the layout from a high angle, with the tape visible if you can.

### The passes

Aim for **12–15**, so ≥10 survive. Vary them deliberately — ten identical passes teach
the evaluation nothing, and R-09 is about not overfitting to a narrow slice.

- [ ] Straight reverse toward the pole, stopping well short
- [ ] Straight reverse toward boxes
- [ ] Reverse with a left curve past an obstacle
- [ ] Reverse with a right curve past an obstacle
- [ ] Reverse parallel to the kerb, close but not touching
- [ ] A pass where the **mirror** is the nearest part to something, not the bumper
- [ ] A pass where a **wheel** is the nearest part (kerb alongside)
- [ ] One or two at very low speed, and one or two slightly faster (still under 5 km/h)
- [ ] A pass with nothing nearby at all — the false-alarm control case

For every pass:

- [ ] **A sharp vertical shake or a hand clap on the phone at the very start and end of
      each recording.** This is the sync signal: it puts a sharp spike in both the IMU
      trace and the image motion, which is what lets the video/IMU offset be estimated
      (P1-T4). Without it, sync is guesswork.
- [ ] Note the pass number and what it was testing, on paper, as you go.
- [ ] **Record which doors and mirrors are open or closed** (R-15) — normally all closed
      for lot runs; say so explicitly.
- [ ] Stop the vehicle before touching the phone.

### After the passes

- [ ] Re-measure two obstacle positions. If anything has moved, record the new position
      and which passes it affects.
- [ ] Play back two clips and check them.

---

## 4. Immediately after any capture day — do not postpone

Raw data is irreplaceable (§0), and this is the step people skip when they are tired.

- [ ] Copy files off the phone to the laptop. **Copy, do not move**, until the checksums
      verify.
- [ ] Run `python scripts/ingest.py --src <path> --kind scan|lot`. This computes sha256,
      writes the row in `data/MANIFEST.md`, and assigns the **dev/test split**.
- [ ] **Three copies total** (laptop, external drive, cloud) and **verify the sha256 after
      each copy, not before** (R-13). A copy that was never verified is not a backup.
- [ ] Only once all three verify: delete from the phone.
- [ ] Write the devlog entry **today**, while you still remember what went wrong. Include
      the hand-measured dimensions, the marker layout, which doors were open, mount
      position, and anything that surprised you.
- [ ] Put `gt.yaml` in the raw folder for lot days.

> **About the dev/test split.** It is assigned at ingest and never changed afterwards
> (R-09). The **test split is opened exactly once, at P5-T2.** Not to "just check". Every
> look costs you the ability to claim the result is honest, and P5-T1/T2 is the one result
> §6 says can never be cut.

---

## 5. Privacy — before anything leaves `data/`

From §9. This applies to the demo video, the README, `docs/evidence/`, and anything shown
to anyone.

- [ ] Recordings may contain **bystanders' faces and licence plates**. Assume they do.
- [ ] Run a face + plate **blur pass**, then **look at the result yourself** frame by
      frame before it leaves `data/`. An automatic blur that missed one plate is not a
      blur pass.
- [ ] **Strip location metadata** from every published image and video.
- [ ] Never commit raw recordings, GPS tracks, or anything that reveals a home address or
      a daily routine. `.gitignore` blocks `data/`; do not work around it.
- [ ] Third-party footage is for **private testing only** and is never republished.
- [ ] Nothing is published without the developer's explicit approval (§0).

---

## 6. Abort and re-shoot triggers

Stop, and plan a re-shoot, rather than pressing on:

- Rain, ice, or poor visibility starts — **stop immediately** (§9)
- The lot stops being empty
- Sun breaks through and the van goes glossy and blown-out (scan day) — wait, or come back
- The phone mount slips mid-session — stop; every pass after the slip has different
  extrinsics, so end the session and restart it as a new one
- Focus or exposure lock releases — stop, re-lock, and note which clips are affected
- An obstacle gets moved and nobody noticed when — the ground truth for the affected
  passes is gone
- Storage or battery runs low — stop cleanly rather than losing the last file mid-write

---

## Open questions

These need a decision from the developer before a capture day is worth booking.

### How to settle both of them in one sitting

Do not try to answer these by reading app descriptions. App capabilities change, and
"the app says it locks exposure" is exactly the kind of claim that turns out to be false
once you are standing in a car park. Measure it instead.

**Step 1 — record a test capture at home** (twenty minutes, no van required):

- [ ] Pick a candidate recording setup: either one app that records video *and* IMU, or
      a video app plus a separate sensor logger running at the same time.
- [ ] Lock focus and exposure, main lens, no zoom — the settings you intend to use on the
      day (R-04).
- [ ] Record roughly **30 seconds walking slowly around a textured object** — a bookshelf
      is ideal, a blank wall is useless.
- [ ] **Shake the phone sharply, or clap against it, at the very start and again at the
      very end.** This is not optional. It is the only unambiguous feature both the video
      and the IMU see at the same instant, and it is what the offset estimate locks onto.
- [ ] Get both files off the phone: the video, and the IMU log as CSV.

**Step 2 — run the validator:**

```bash
python scripts/check_capture.py --video test.mp4 --imu gyro.csv
```

It reports, per check, whether the timestamps are usable, whether the IMU log parses and
at what rate, and whether the video/IMU offset can actually be recovered and with what
confidence. Exit code is 0 unless something failed, so it can gate a capture day.

**Step 3 — the stabilisation check**, once the camera has been calibrated (P0-T7):

- [ ] Shoot about 20 seconds of the checkerboard **while walking slowly**.
- [ ] `python scripts/check_capture.py --video test.mp4 --imu gyro.csv --moving-board board_walk.mp4`

This compares the reprojection error of the calibrated intrinsics on a still board against
a moving one. A rigid lens keeps roughly the same error either way; if the frames are being
warped per frame, the error jumps. Note that this **cannot separate electronic stabilisation
from rolling shutter**, and does not try to — both break the pinhole model the same way and
call for the same response.

Rerun the whole thing on the morning of the capture as a go/no-go. It takes a minute and it
is the cheapest insurance in the project.


### 1. How is IMU data recorded alongside video? *(blocks P1-T3)*

P1-T3 requires "video + IMU" and P4-T2 needs visual-inertial ego-motion. **A phone's
stock camera app does not log IMU.** Something has to, with timestamps that can be related
to the video. Roughly:

- a single app that records video **and** a sensor log together (cleanest, if one exists
  for this phone that also allows locked focus/exposure);
- stock camera plus a **separate sensor-logger app** running simultaneously — two files,
  two clocks, and the offset is what `capture/sync.py` estimates from the shake spike;
- **no IMU**, accepting the §8.2 fallback to visual-only odometry from the start.

Whatever is chosen, it must be **tested at home before capture day**: record 30 seconds,
confirm both files exist, and confirm the IMU log has real timestamps. Finding this out in
the lot is a wasted day.

### 2. Can stabilisation actually be turned off?

If EIS cannot be disabled, it warps frames individually and the calibration no longer
describes the images. Worth ten minutes to check now. If it cannot be turned off, record
that as a known limitation and expect it to be the first suspect if reconstruction is poor.

---

*Reviewed by developer:* **approved** — *date:* **2026-09-24**

> **Approving this document did not answer the two open questions above**, and those still
> block booking a capture day:
>
> 1. **How IMU gets recorded alongside video.** Nothing does it yet, and P1-T3 requires
>    video + IMU. Needs an app choice *and a 30-second test at home* confirming both files
>    exist with real timestamps.
> 2. **Whether stabilisation can be disabled.** Ten minutes to check.
>
> Neither is a document problem, so neither is fixed by signing the document.
