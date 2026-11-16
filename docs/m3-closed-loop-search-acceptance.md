# M3 — Closed-loop Search Engineering Acceptance (v1)

Date: 2026-11-16  
Branch: `feat/closed-loop-search`  
Commit: `0539f3e64e21f44048ee7cde4f586520be89ab7c`  
Result: **PASS** — camera calibration and real-environment
`VisualSearchPolicy` integration.

This is an engineering acceptance run. It uses scripted detections, not a
YOLO detector. No `best.pt` was available or used, no VPT/HH1/HH2 training was
run, and no Ray cluster was started. The test uses one real Minecraft
environment and the existing 2M Flat PPO checkpoint.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D, 24,564 MiB |
| GPU driver | 595.84 |
| Python | 3.10.22 |
| PyTorch | 2.5.1+cu124, CUDA available |
| MineStudio | 1.1.6 |
| NumPy | 1.26.4 |
| Java | OpenJDK 1.8.0_504 |
| Display | Xvfb `:99` |
| Ray | not started |
| 2M VPT checkpoint | `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/model.ckpt` |
| Checkpoint SHA256 | `d961b9889f1a2424569bebcaa2633589e17aa6fc229c5e3b6283c0e2aa924f79` |
| `yaw_sign` | **1** |

The full machine-readable record is in
`experiments/search_interface_smoke/acceptance.json` and
`experiments/search_interface_smoke/environment.txt`.

## Task A — Real camera calibration

Command:

```bash
python scripts/calibrate_search_camera.py \
  --config configs/env/zombie_stage1_random_face.yaml \
  --degrees 8 \
  --steps 3 \
  --output-dir experiments/search_camera_calibration_v1 \
  2>&1 | tee experiments/search_camera_calibration_v1.log
```

### Action encoding

| Direction | Requested | Button ID | Camera ID | Decoded pitch | Decoded yaw | Extra buttons |
|---|---:|---:|---:|---:|---:|---|
| positive | +8° | 1 | 65 | 0° | +10° | none |
| negative | -8° | 1 | 55 | 0° | -10° | none |

Both requested angles became valid discrete camera actions. The 8° request is
quantized to the environment's 10° camera bins; direction and ordering are
preserved. The calibration script also decoded each action and rejected any
non-camera button, so there was no unintended attack, movement, or inventory
action.

### Visual direction

The four required screenshots are preserved in the shared delivery under
`search_acceptance_v1/calibration/` (the repository keeps `experiments/**/*.png`
ignored by design). The two `before` frames were captured before the
asynchronously spawned Zombie finished rendering, so they do not show the
target clearly. To avoid guessing the sign, a supplementary settled-target
trajectory was captured with the same stationary-Zombie setup:

- positive yaw, target body center x (128 px frame): `64.5 → 55.5 → 45.5 → 35.0`
  px — moves **left**;
- negative yaw, target body center x: `62.5 → 71.5 → 81.5 → 92.5` px —
  moves **right**.

The official `after_positive.png` / `after_negative.png` show the same result:
the target ends left of center for positive yaw and right of center for negative
yaw. Per the task convention, positive yaw moving the target left means:

```text
yaw_sign = 1
```

Supplementary evidence:
`search_acceptance_v1/direction/direction_evidence.png`.

**Task A: PASS.**

## Task B — Real-environment VisualSearchPolicy integration

The real Minecraft environment and real 2M VPT policy were used with the
scripted 9-detection sequence from the task. SEARCH and ALIGN executed the
camera adapter; COMBAT executed VPT. The observed output was:

```text
step=1 mode=search reward=-0.0010
step=2 mode=align reward=-0.0010
step=3 mode=combat reward=-0.0010
step=4 mode=combat reward=-0.0010
step=5 mode=combat reward=-0.0010
step=6 mode=search reward=-0.0010
step=7 mode=align reward=-0.0010
step=8 mode=combat reward=-0.0010
step=9 mode=combat reward=-0.0010
```

The observed mode sequence exactly matched the expected sequence. The VPT call
trace was `[True, False, False, True, False]`, meaning:

- VPT was called only in the five COMBAT steps;
- the first COMBAT entry initialized a fresh VPT memory;
- VPT memory was carried through consecutive COMBAT steps;
- after returning to SEARCH/ALIGN, the next COMBAT entry initialized a fresh
  memory again.

`SearchTracePolicy` recorded the run:

```text
steps=9 search_steps=2 align_steps=2 combat_steps=5
detected_steps=5 first_detection_step=2 first_combat_step=3
reacquisitions=1
```

The log contains no OOM, no traceback, no Minecraft crash, and no fatal
exception. The remaining error lines are pre-existing non-fatal audio/narrator
and OptiFine texture warnings. `env.close()` completed and no Minecraft engine
process remained afterward; the GPU returned to idle.

Full log: `experiments/search_interface_smoke/run.log`.

**Task B: PASS.**

## Acceptance checklist

| Check | Result |
|---|---|
| Branch contains `0539f3e` or newer | PASS (`0539f3e`) |
| Continuous yaw encoded as valid discrete camera action | PASS |
| No extra attack/movement/inventory buttons | PASS |
| Camera direction verified from frames | PASS (`yaw_sign=1`) |
| Real Minecraft accepts all 9 steps | PASS |
| Real VPT inference in COMBAT | PASS |
| VPT not called in SEARCH/ALIGN | PASS |
| VPT memory reset on COMBAT re-entry | PASS |
| SearchTrace records the episode | PASS |
| Environment closes cleanly | PASS |
| OOM / Minecraft crash | none |
| YOLO `best.pt` used | no (controlled detections) |

## Delivery

Shared root:

```text
/eaas/default/groups/mine_/home/share/minecombat_results/search_acceptance_v1/
```

The shared package contains:

```text
calibration/calibration.json
calibration/before_positive.png
calibration/after_positive.png
calibration/before_negative.png
calibration/after_negative.png
calibration/search_camera_calibration_v1.log
integration/run.log
integration/environment.txt
integration/acceptance.json
direction/direction_evidence.png
README.md
```

Small records are also committed to GitHub. Raw artifacts remain in the shared
directory and are not part of the training data or detector dataset.

## Next

The closed-loop search engineering chain is accepted for this stage. Do not
start 360° search performance evaluation until the annotation pass and a real
Zombie detector (`best.pt`) with validation metrics are available.
