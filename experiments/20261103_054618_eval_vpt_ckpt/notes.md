# Camera entropy A/B — ±60° jitter (50 seeds)

Checkpoint: `experiments/flat_100k_mixed_angle60_camera_entropy/checkpoints/2026-11-03_05-10-32/2/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle60_mask.yaml`  
Training overrides: `--entropy-coef 0.01 --camera-entropy-scale 5`  
Eval config: `configs/env/zombie_stage1_easy_mixed_angle60_mask.yaml`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.10 death_rate=0.64 timeout_rate=0.26
mean_hp_lost=16.44 mean_hp_lost_on_win=11.40
mean_steps=373.00 mean_time_to_kill=240.00 mean_reward=-2.28
```

The ±60° target moved only from 8% to 10% (within noise). Camera entropy did not
solve the off-screen search problem at this scale.
