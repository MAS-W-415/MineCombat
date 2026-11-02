# Mixed + angle60 curriculum 100k — ±60° jitter (50 seeds)

Checkpoint: `experiments/flat_100k_mixed_angle60_mask_warm/checkpoints/2026-11-03_04-22-35/2/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle60_mask.yaml`  
Eval config: `configs/env/zombie_stage1_easy_mixed_angle60_mask.yaml`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.08 death_rate=0.74 timeout_rate=0.18
mean_hp_lost=16.50 mean_hp_lost_on_win=12.00
mean_steps=345.00 mean_time_to_kill=221.50 mean_reward=-2.45
```

The target task collapsed to 8% win / 74% deaths. At ±60° many spawns are outside
the field of view, so the policy cannot see the target at reset. It appears to
learn a local optimum: wait for the zombie to enter the view instead of turning
to search. This is the off-screen search problem, not a distance or defense
problem.
