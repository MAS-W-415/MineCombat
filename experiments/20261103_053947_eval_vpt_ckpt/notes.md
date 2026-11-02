# Camera entropy A/B — easy front (50 seeds)

Checkpoint: `experiments/flat_100k_mixed_angle60_camera_entropy/checkpoints/2026-11-03_05-10-32/2/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle60_mask.yaml`  
Training overrides: `--entropy-coef 0.01 --camera-entropy-scale 5`  
Eval config: `configs/env/zombie_stage1_easy.yaml`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.36 death_rate=0.60 timeout_rate=0.04
mean_hp_lost=14.34 mean_hp_lost_on_win=8.50
mean_steps=250.00 mean_time_to_kill=218.06 mean_reward=+1.38
```

Easy front fell from 58% (no entropy) to 36%. Camera exploration at this scale
disrupts front-facing combat more than it helps search.
