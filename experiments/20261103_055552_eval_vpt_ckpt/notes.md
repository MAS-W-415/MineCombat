# Camera entropy A/B — random spawn (50 seeds)

Checkpoint: `experiments/flat_100k_mixed_angle60_camera_entropy/checkpoints/2026-11-03_05-10-32/2/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle60_mask.yaml`  
Training overrides: `--entropy-coef 0.01 --camera-entropy-scale 5`  
Eval config: `configs/env/zombie_stage1.yaml`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.02 death_rate=0.74 timeout_rate=0.24
mean_hp_lost=16.14 mean_hp_lost_on_win=9.00
mean_steps=403.00 mean_time_to_kill=248.00 mean_reward=-3.21
```

Random spawn fell from 4% to 2%. Reject camera entropy at this scale. The
off-screen search problem needs a different mechanism or a separate gate.
