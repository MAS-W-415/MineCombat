# M3 — Search Dataset Collection v1

Date: 2026-11-16  
Branch: `feat/closed-loop-search` (`7b0edc2`)  
Checkpoint: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/`  
Checkpoint SHA256: `d961b9889f1a2424569bebcaa2633589e17aa6fc229c5e3b6283c0e2aa924f79`  
GPU: NVIDIA RTX 4090 D 24 GB  
MineStudio 1.1.6, Python 3.10.22, PyTorch 2.5.1+cu124, JDK 8u504

## Goal

Collect real Minecraft RGB frames for a future Zombie detector used by the
closed-loop search controller. Two distributions:

- **face**: random distance/360° spawn, but the player faces the zombie at reset.
- **random**: full random 360° spawn.

This is data collection only; no VPT or HH PPO training was run.

## Commands

```bash
# A group
python scripts/record_eval.py \
  --checkpoint "$CKPT" \
  --config configs/env/zombie_stage1_random_face.yaml \
  --episodes 10 --seed-start 2000 --max-steps 300 --deterministic \
  --save-policy-frames --frame-stride 10 \
  --output-dir experiments/search_dataset_face_v1

# B group
python scripts/record_eval.py \
  --checkpoint "$CKPT" \
  --config configs/env/zombie_stage1.yaml \
  --episodes 10 --seed-start 3000 --max-steps 300 --deterministic \
  --save-policy-frames --frame-stride 10 \
  --output-dir experiments/search_dataset_random_v1
```

## Results

| Group | Episodes | PNG | Videos | Wall time | Outcomes |
|---|---:|---:|---:|---:|---|
| face | 10 | 255 | 10 | 260.81 s | success 5, death 2, ongoing 3 |
| random | 10 | 305 | 10 | 276.59 s | success 0, death 2, ongoing 8 |
| **total** | 20 | **560** | 20 | **537.40 s** | |

Face group per-seed frames: 2000=14, 2001=31, 2002=22, 2003=31, 2004=28,
2005=27, 2006=31, 2007=23, 2008=26, 2009=22.

Random group: 3000=29, 3001=28, the other eight seeds=31.

## Quality checks

| Check | Result |
|---|---|
| PNG count vs manifest | 255/255 and 305/305 |
| Resolution | all 128×128 |
| Channels | all RGB |
| Missing files | 0 |
| Duplicate frames | 0 |
| Black frames | 0 |
| Manifest path/step alignment | yes |
| Environment crashes | 0 |

Visual spot check (contact sheets in the shared `previews/` directory):

- face: centered targets, edge/far targets, close targets, combat frames,
  partially occluded frames, and no-target frames.
- random: many no-target/timeout frames (useful negatives), a few centered/far
  targets, one close target, and target-loss frames.
- In the face group, some step-0 frames are captured before the target is
  rendered; later frames contain visible targets. Annotators should prefer
  non-zero steps or verify step-0 frames.

## Delivery layout

Shared root:

```text
/eaas/default/groups/mine_/home/share/minecombat_results/search_dataset_v1/
├── face/
│   ├── policy_frames/            # 255 PNG + manifest.jsonl
│   └── manifest.json             # 10 video records
├── random/
│   ├── policy_frames/            # 305 PNG + manifest.jsonl
│   └── manifest.json
├── videos/
│   ├── face/                     # 2000, 2004, 2002, 2001, 2006
│   └── random/                   # 3000, 3001, 3008, 3002, 3005
├── samples/
│   ├── face/                     # 20 representative PNGs, seed/step paths
│   └── random/                   # 20 representative PNGs
└── previews/                     # contact sheets
```

Raw PNGs and MP4s stay in the shared directory, not in Git. GitHub contains the
manifests, statistics, and this acceptance record.

## Videos

Face:

- `2000`: success
- `2004`: success
- `2002`: death
- `2001`: ongoing / target loss
- `2006`: ongoing / close target

Random:

- `3000`: death
- `3001`: death
- `3008`: close target / engagement
- `3002`: ongoing / no target
- `3005`: ongoing / target loss

The random group had no success episode in this sample; the selected videos
cover close engagement, death, no-target, and target-loss cases.

## Acceptance checklist

- [x] Real 2M Flat PPO checkpoint
- [x] Face and random 360° collections both completed
- [x] PNG and manifest counts aligned
- [x] Images 128×128 RGB, no black/duplicate/missing frames
- [x] Visible-target and no-target candidates provided
- [x] Representative videos and 40 sample previews provided
- [x] Config, versions, checkpoint, and collection statistics recorded
- [x] Data location shared with the team

## Next

1. Review the 40 samples and contact sheets; confirm annotation conventions.
2. Select 100–200 frames from different episodes for CVAT.
3. Build the YOLO dataset with `dataset_builder.py`.
4. Train the first lightweight detector with `train_detector.py`; evaluate
   Precision/Recall/mAP.
5. Integrate the detector with `ClosedLoopSearchController` and re-run random
   spawn evaluation.
