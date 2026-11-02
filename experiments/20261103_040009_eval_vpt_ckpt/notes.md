# Mixed + angle curriculum 100k — ±30° jitter (50 seeds)

Checkpoint: `experiments/flat_100k_mixed_angle30_mask_warm/checkpoints/2026-11-03_03-25-05/2/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Eval config: `configs/env/zombie_stage1_easy_angle30_mask.yaml`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.34 death_rate=0.52 timeout_rate=0.14
mean_hp_lost=13.66 mean_hp_lost_on_win=7.76
mean_steps=288.00 mean_time_to_kill=198.53 mean_reward=+1.21
```

The main result: ±30° win rate rose from 10% (far checkpoint) and 16% (pure
angle 100k) to 34%. The 24-point gain over the far checkpoint is about 3.5
standard errors at 50 seeds.
