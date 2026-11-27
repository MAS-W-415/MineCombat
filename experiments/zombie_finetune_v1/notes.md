# Zombie detector fine-tune annotation — v1

Date: 2026-11-22

First Train/Val annotation batch for the Mob Radar YOLO26 Nano fine-tune.

```text
images:        40
positive:      27
negative:      13
boxes:         27
train:         32 (23 positive / 9 negative)
val:            8 (4 positive / 4 negative)
```

The split is by episode:

```text
train seeds: 2000, 2002, 2003, 2004, 2005, 2006, 3000, 3002
val seeds:   2001, 3001
```

Annotation was done in CVAT with one class (`0: zombie`). One duplicate auto box
on `seed_2005_step_00090` was removed; no tracks remain. All negative frames
were reviewed as true background frames.

YOLO label files are committed under `labels/train/` and `labels/val/`.
Images were copied to the GPU host for training.

## Fine-tune v1

A 20-epoch single-class fine-tune from `mob_radar_best.pt` completed. The
5-class Zombie head was remapped to the 1-class head (708/708 items
transferred). Fine-tuned `best.pt` SHA256:

```text
23cdd80a89e7a4d33e217dea315833212af1dbb80c66d294f50449b13eb82b37
```

Pilot audit (80 images, not an independent test): recall improved from 56.4%
to 82.1%, FPR remained 0%.

An independent CVAT task `test_40` (40 new images, two jobs, 7 prelabels) was
created and is awaiting annotation.

See `docs/mob-radar-finetune-v1.md` and `docs/annotation-status.md`.
