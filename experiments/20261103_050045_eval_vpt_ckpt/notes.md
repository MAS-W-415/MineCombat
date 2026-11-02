# Mixed + angle60 curriculum 100k — mixed distance (20 seeds)

Checkpoint: `experiments/flat_100k_mixed_angle60_mask_warm/checkpoints/2026-11-03_04-22-35/2/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle60_mask.yaml`  
Eval config: `configs/env/zombie_stage1_easy_mixed_mask.yaml`  
Seeds: 0–19, max 600 steps, deterministic

```text
win_rate=0.45 death_rate=0.55 timeout_rate=0.00
mean_hp_lost=13.90 mean_hp_lost_on_win=8.67
mean_steps=250.00 mean_time_to_kill=220.22 mean_reward=+2.43
```

20-seed screen only. Mixed-distance performance is essentially unchanged from
the ±30 checkpoint (48% at 50 seeds).
