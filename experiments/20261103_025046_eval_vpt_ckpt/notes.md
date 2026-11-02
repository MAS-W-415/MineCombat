# Angle curriculum 100k — mixed distance 3–7 (20 seeds)

Checkpoint: `experiments/flat_100k_angle30_mask_warm/checkpoints/2026-11-03_02-11-52/2/`  
Training config: `configs/env/zombie_stage1_easy_angle30_mask.yaml`  
Eval config: `configs/env/zombie_stage1_easy_mixed_mask.yaml`  
Seeds: 0–19, max 600 steps, deterministic

```text
win_rate=0.30 death_rate=0.65 timeout_rate=0.05
mean_hp_lost=15.50 mean_hp_lost_on_win=10.00
mean_steps=280.00 mean_time_to_kill=250.00 mean_reward=+0.86
```

Distance generalization regressed from the far checkpoint's zero-shot 40% to
30%.
