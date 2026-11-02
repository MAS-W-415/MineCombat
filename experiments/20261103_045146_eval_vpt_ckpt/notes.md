# Mixed + angle60 curriculum 100k — easy front (50 seeds)

Checkpoint: `experiments/flat_100k_mixed_angle60_mask_warm/checkpoints/2026-11-03_04-22-35/2/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle60_mask.yaml`  
Eval config: `configs/env/zombie_stage1_easy.yaml`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.58 death_rate=0.42 timeout_rate=0.00
mean_hp_lost=12.42 mean_hp_lost_on_win=8.38
mean_steps=209.00 mean_time_to_kill=197.72 mean_reward=+4.34
```

Easy front reached 58%, the best close-range result so far. The wider training
distribution appears to have taught a stronger wait-and-punish behavior when the
zombie comes into view.
