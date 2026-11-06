# 2M flat PPO continuation

Source: `experiments/flat_1m_mixed_angle30_4090d/checkpoints/2026-11-06_08-05-32/20/`  
Config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Code: `b54c53d`  
Hardware: RTX 4090 D (24 GB), 60 GB RAM

```text
environment steps: 1,024,000 (2.048M total)
iterations: 20
wall time: 5,603.46 s (93.4 min)
peak GPU: 6,039 MiB
peak RAM: 39 GB
NaN/Inf: 0
too-high-KL warnings: 4,572
```

Checkpoints retained with `keep_interval=5`:

```text
checkpoints/2026-11-06_11-45-28/5/   (1.25M total)
checkpoints/2026-11-06_11-45-28/10/  (1.50M total)
checkpoints/2026-11-06_11-45-28/15/  (1.75M total)
checkpoints/2026-11-06_11-45-28/20/  (2.00M total)
```

W&B was offline; upload failed with HTTP 403 (no API key). The offline run is
preserved under `experiments/wandb_offline_2m/` and is not synced.

Evaluation: `experiments/eval_2m_4090d/` and `docs/m3-2m-continuation.md`.
