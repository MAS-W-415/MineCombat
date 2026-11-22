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
Images are staged in a local archive and will be copied to the GPU host or
shared storage before training. The independent Test set and the actual
fine-tune are not part of this record yet.

See `docs/annotation-status.md` for the full acceptance record.
