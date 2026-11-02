# Angle curriculum 100k — easy front (50 seeds)

Checkpoint: `experiments/flat_100k_angle30_mask_warm/checkpoints/2026-11-03_02-11-52/2/`  
Training config: `configs/env/zombie_stage1_easy_angle30_mask.yaml`  
Eval config: `configs/env/zombie_stage1_easy.yaml`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.44 death_rate=0.56 timeout_rate=0.00
mean_hp_lost=14.18 mean_hp_lost_on_win=9.27
mean_steps=243.00 mean_time_to_kill=211.50 mean_reward=+2.70
```

Close-range retention dropped from the far checkpoint's 52% win / 40% death to
44% / 56%, but stayed above the 40% keep threshold.
