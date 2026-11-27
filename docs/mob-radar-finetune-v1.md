# Mob Radar Zombie Detector Fine-tune v1

Date: 2026-11-27  
Base weights: `mob_radar_best.pt` (YOLO26 Nano, 5 classes)  
Fine-tuned weights: `zombie_finetune_v1/training/finetune_20ep/weights/best.pt`  
Fine-tuned SHA256: `23cdd80a89e7a4d33e217dea315833212af1dbb80c66d294f50449b13eb82b37`

## Data

```text
Train/Val images: 40
Positive frames:  27
Negative frames:  13
Boxes:            27
Train:            32 (23 positive / 9 negative)
Val:               8 (4 positive / 4 negative)
```

Split by episode:

```text
Train: 2000, 2002, 2003, 2004, 2005, 2006, 3000, 3002
Val:   2001, 3001
```

One duplicate auto box (`seed_2005_step_00090`) was removed. The final labels
have no tracks and all negative files are empty.

## Training

```text
epochs:        20
imgsz:         320
batch:          8
seed:          42
device:        RTX 4090 D, CUDA 0
training time: 0.005 h
```

The 5-class checkpoint was transferred to a single-class `zombie` head:

```text
Remapped 1/1 cls head rows from pretrained weights by class name
Transferred 708/708 items
```

## Validation

On the 8-image Val split (only 4 ground-truth boxes):

| Metric | Value |
|---|---:|
| Precision | 0.972 |
| Recall | 0.500 |
| mAP50 | 0.495 |
| mAP50-95 | 0.371 |

The Val set is too small for a reliable final metric; these numbers are only a
sanity check and early-stopping signal.

## Pilot comparison on the previous 80-image audit

The previous zero-shot audit set was re-used for a **pilot comparison only**.
It is not the independent test set. The fine-tuned model was trained on 40
different frames, but some audit frames share source episodes with training.

| Model | TP | FN | FP | TN | Frame Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Original Mob Radar | 22 | 17 | 0 | 41 | 56.4% | 0.0% |
| Fine-tuned v1 | **32** | **7** | 0 | 41 | **82.1%** | 0.0% |

Per group:

| Model | Face Recall | Random Recall |
|---|---:|---:|
| Original | 62.5% | 28.6% |
| Fine-tuned | **84.4%** | **71.4%** |

This is a strong pilot signal, but **not a final independent-test result**.

## Fine-tuned misses on the pilot audit

The remaining 7 misses:

```text
extra_seed_2007_step_00180.png
seed_2002_step_00110.png
seed_2006_step_00300.png
seed_2007_step_00220.png
seed_2009_step_00110.png
extra_seed_3000_step_00170.png
extra_seed_3008_step_00030.png
```

They are mostly extreme close-up, sunset-lit, or very small/edge targets.

## Independent Test Task

A separate CVAT task was created with 40 new images from unseen episodes:

```text
Task: test_40
Task ID: 2796389
Images: 40
  Face positives: 20
  Random negatives: 20
Jobs: 2
  Job 4772812: frame 0–19
  Job 4772813: frame 20–39
Prelabels: 7
Status: awaiting annotation
```

The test task is under `MineCombat` and is deliberately separate from Train/Val.
Test labels must be completed and frozen before the final comparison.

## Decision

The fine-tune is promising enough to continue to the independent test stage.

Do **not** claim a final mAP improvement or 360° search improvement until:

1. `test_40` is annotated and frozen.
2. Original and fine-tuned weights are evaluated on the same test labels.
3. Precision / mAP50 / mAP50-95 are reported.
4. A small closed-loop smoke is run if the test metrics are acceptable.

No VPT, HH1, HH2, or large 360° PPO evaluation was run in this step.
