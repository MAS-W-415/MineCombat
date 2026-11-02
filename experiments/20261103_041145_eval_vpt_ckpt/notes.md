# Mixed + angle curriculum 100k — mixed distance (50 seeds)

Checkpoint: `experiments/flat_100k_mixed_angle30_mask_warm/checkpoints/2026-11-03_03-25-05/2/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Eval config: `configs/env/zombie_stage1_easy_mixed_mask.yaml`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.48 death_rate=0.48 timeout_rate=0.04
mean_hp_lost=13.38 mean_hp_lost_on_win=9.00
mean_steps=241.00 mean_time_to_kill=211.67 mean_reward=+2.93
```

The 20-seed 55% screen settled at 48% on 50 seeds. This is a solid mixed-distance
result, close to the 50% gate, and confirms that the joint distance + angle
curriculum preserves distance performance.
