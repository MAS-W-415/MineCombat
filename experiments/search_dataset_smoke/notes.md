# Search dataset smoke

Date: 2026-11-16  
Checkpoint: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/`  
Config: `configs/env/zombie_stage1_random_face.yaml`  
Code: `e199939` (PR #68 search interfaces)  
Hardware: local RTX 4060, 31 GB RAM

## Command

```bash
python scripts/record_eval.py \
  --checkpoint experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/model.ckpt \
  --config configs/env/zombie_stage1_random_face.yaml \
  --episodes 2 \
  --max-steps 100 \
  --seed-start 1000 \
  --deterministic \
  --save-policy-frames \
  --frame-stride 5 \
  --output-dir experiments/search_dataset_smoke
```

## Outputs

- 2 MP4 videos: `episode_1000.mp4`, `episode_1001.mp4`
- 42 PNG policy frames: 21 per seed (steps 0,5,…,100)
- `policy_frames/manifest.jsonl`: 42 samples
- `manifest.json`: video manifest

Both episodes ended at the 100-step limit with `outcome=ongoing`, as expected.
No PPO update was run.

Video and PNG files are intentionally not committed (`.gitignore`); this record
keeps the manifests and the notes. The sample frame at step 50 shows the zombie
in front, confirming that `random_face` worked.

## Purpose

Validate the new PR #68 data path:

- `record_eval.py --save-policy-frames`
- `PolicyFrameExporter`
- per-step manifest/PNG naming for later YOLO annotation

This is a data-collection smoke, not a policy performance result.
