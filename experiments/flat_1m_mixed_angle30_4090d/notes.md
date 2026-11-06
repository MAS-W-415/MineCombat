# 1M flat PPO — mixed + angle curriculum

Source: `experiments/flat_100k_mixed_angle30_mask_warm/checkpoints/2026-11-03_03-25-05/2/`  
Config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Code: `b54c53d`  
Hardware: RTX 4090 D (24 GB), 60 GB RAM, Ubuntu 24.04

```text
environment steps: 1,024,000
iterations: 20
wall time: 4,673.81 s (77.9 min)
peak GPU: 6,035 MiB
peak RAM: 40 GB
NaN/Inf: 0
too-high-KL warnings: 2,150
```

Final checkpoint:

```text
experiments/flat_1m_mixed_angle30_4090d/checkpoints/2026-11-06_08-05-32/20/model.ckpt
SHA256: 830f9c4038af5b6ddd7e5ef4fee37f80cfce47230b3727c4fc6b089faaebf8b4
```

The checkpoint is not committed to Git; it is stored on the remote host and in
the group share (`minecombat_results/flat_1m_mixed_angle30_4090d/`).

Intermediate 5/10/15 checkpoints were deleted by `keep_interval=40`. The
continuation run uses `--keep-interval 5`.

Evaluation: `experiments/eval_1m_4090d/` and `docs/m3-1m-flat-ppo.md`.
