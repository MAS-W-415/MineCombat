# Zombie Detector Annotation Status

Date: 2026-11-22  
Branch: `main`  
Scope: CVAT annotation for the first zombie-detector fine-tuning batch.  
No VPT, HH1, HH2, or detector training was run in this step.

## Result

The first Train/Val annotation batch is complete.

| Item | Value |
|---|---:|
| Total annotated images | 40 |
| Positive frames | 27 |
| Negative frames | 13 |
| Bounding boxes | 27 |
| Train images | 32 |
| Train positive / negative | 23 / 9 |
| Val images | 8 |
| Val positive / negative | 4 / 4 |

Label format:

```text
0 center_x center_y width height
```

Only class `0: zombie` is used. Negative frames have empty `.txt` files.

## Episode split

The split is by Episode, never by individual image.

```text
Train seeds:
  2000, 2002, 2003, 2004, 2005, 2006, 3000, 3002

Val seeds:
  2001, 3001
```

The split was chosen after annotation to keep positive examples in both
partitions. It differs from the initial draft split because the actual positive
counts per episode were not known until annotation was complete.

## CVAT workflow

Two CVAT tasks were used:

```text
trainval_calibration_10      10 images, completed
trainval_remaining_30        30 images, completed
  Job A: frame 0–14
  Job B: frame 15–29
```

The 10-image calibration task was a personal-workspace pilot. The 30-image
batch was created in the `MineCombat` organization/project. Both tasks are
completed.

## QA performed

- Removed one duplicate auto box on `seed_2005_step_00090`; the manual box
  remains.
- No tracks remain in the final annotations.
- The six Face empty frames were reviewed and are true negatives:
  `seed_2001_step_00090`, `seed_2003_step_00210`,
  `seed_2003_step_00240`, `seed_2006_step_00090`,
  `seed_2006_step_00210`, `seed_2006_step_00240`.
- The Random empty frames were reviewed and are true negatives.
- All 40 images have exactly one label file; negative label files are empty.
- `data.yaml` uses one class, `0: zombie`.
- `split_manifest.json` records seed, step, split, and box count for every image.

## Dataset staging

The assembled dataset was staged locally as:

```text
/tmp/opencode/zombie_finetune_v1/dataset/
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml
  summary.json
  split_manifest.json
```

A portable local archive with images and labels is also available at:

```text
/home/mas/desktop/MineCombat_zombie_finetune_v1_trainval.zip
```

The YOLO label text files are committed under
`experiments/zombie_finetune_v1/labels/`; PNG images are not committed.
The images and labels still need to be copied to the GPU host or shared storage
before training.

## Fine-tune v1

The staged dataset was copied to the GPU host and used for a 20-epoch
single-class fine-tune from `mob_radar_best.pt`.

```text
fine-tuned checkpoint SHA256:
23cdd80a89e7a4d33e217dea315833212af1dbb80c66d294f50449b13eb82b37

validation (8 images, 4 boxes):
precision 0.972, recall 0.500, mAP50 0.495, mAP50-95 0.371

pilot audit (80 images):
original recall 56.4% -> fine-tuned 82.1%, FPR 0%
```

The 80-image audit is **not** the independent Test set. Details:
`docs/mob-radar-finetune-v1.md`.

## Independent Test task

A separate CVAT task was created for 40 new images from unseen episodes:

```text
Task: test_40
Task ID: 2796389
Jobs: 4772812 (frame 0–19), 4772813 (frame 20–39)
Images: 20 Face positives + 20 Random negatives
Prelabels: 7
Status: awaiting annotation
```

## Not done yet

- The independent Test task is not annotated or frozen.
- Final old-vs-fine-tuned Precision/mAP comparison is not available.
- No 360° search performance evaluation has been run.

## Next

1. Assign and annotate the two `test_40` jobs.
2. Freeze the Test labels.
3. Evaluate original vs fine-tuned weights on the same Test set.
4. If the test result is acceptable, run a small closed-loop search smoke.

See `docs/cvat-annotation-guide.md` for the annotation procedure and
`docs/m3-mob-radar-zeroshot.md` for the zero-shot baseline.
