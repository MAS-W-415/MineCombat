# Camera entropy A/B — mixed distance (20 seeds)

Checkpoint: `experiments/flat_100k_mixed_angle60_camera_entropy/checkpoints/2026-11-03_05-10-32/2/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle60_mask.yaml`  
Training overrides: `--entropy-coef 0.01 --camera-entropy-scale 5`  
Eval config: `configs/env/zombie_stage1_easy_mixed_mask.yaml`  
Seeds: 0–19, max 600 steps, deterministic

```text
win_rate=0.45 death_rate=0.40 timeout_rate=0.15
mean_hp_lost=10.35 mean_hp_lost_on_win=3.67
mean_steps=295.00 mean_time_to_kill=181.44 mean_reward=+3.15
```

20-seed screen. Win rate is unchanged, but HP lost on wins (3.67) is the lowest
observed. Not enough to justify the camera-entropy direction.
