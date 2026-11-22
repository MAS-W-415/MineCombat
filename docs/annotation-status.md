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

## Not done yet

- The independent Test set has not been collected or annotated.
- The fine-tuned YOLO model has not been trained.
- No 360° search performance evaluation has been run.

## Next

1. Copy the staged dataset to the GPU host.
2. Run a 1-epoch transfer smoke test from `mob_radar_best.pt`.
3. Run the planned 20-epoch, single-class fine-tune.
4. Collect and annotate a separate 40-image Test set.
5. Evaluate old vs fine-tuned weights on the same independent Test set.

See `docs/cvat-annotation-guide.md` for the annotation procedure and
`docs/m3-mob-radar-zeroshot.md` for the zero-shot baseline.
