# Mixed + angle60 curriculum 100k — random spawn (50 seeds)

Checkpoint: `experiments/flat_100k_mixed_angle60_mask_warm/checkpoints/2026-11-03_04-22-35/2/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle60_mask.yaml`  
Eval config: `configs/env/zombie_stage1.yaml`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.04 death_rate=0.66 timeout_rate=0.30
mean_hp_lost=15.24 mean_hp_lost_on_win=12.00
mean_steps=412.00 mean_time_to_kill=235.50 mean_reward=-2.80
```

Random spawn is unchanged at 2/50 wins. Widening the training angle from ±30°
to ±60° did not improve the full 360° task.
