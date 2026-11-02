# Joint ±30 checkpoint — combined task (50 seeds)

Checkpoint: `experiments/flat_100k_mixed_angle30_mask_warm/checkpoints/2026-11-03_03-25-05/2/`  
Eval config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Task: distance 3–7 + ±30° jitter (the training distribution)  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.10 death_rate=0.76 timeout_rate=0.14 stuck_rate=0.14
mean_hp_lost=16.08 mean_hp_lost_on_win=8.40
mean_steps=332.82 mean_time_to_kill=232.20 mean_reward=-2.25
```

This is the key composition result. On the separate tasks the same checkpoint
scores 34% (±30°, distance 3–4) and 48% (mixed distance, front), but on the
combination it scores only 10% with 76% deaths.

The policy does not compose "approach a distant target" with "turn to an
off-center target". Separate-task evaluations overstate the policy's ability on
the joint distribution.
