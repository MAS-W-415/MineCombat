# Angle curriculum 100k — ±30° jitter (50 seeds)

Checkpoint: `experiments/flat_100k_angle30_mask_warm/checkpoints/2026-11-03_02-11-52/2/`  
Training config: `configs/env/zombie_stage1_easy_angle30_mask.yaml`  
Eval config: `configs/env/zombie_stage1_easy_angle30_mask.yaml`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.16 death_rate=0.78 timeout_rate=0.06
mean_hp_lost=16.74 mean_hp_lost_on_win=11.25
mean_steps=289.00 mean_time_to_kill=232.00 mean_reward=-1.37
```

The target task improved only from 10% to 16% win. Of 39 deaths, 21 dealt no
damage, so the policy still often fails to engage an off-center target.
