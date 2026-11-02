# Mixed + angle curriculum 100k — random spawn (50 seeds)

Checkpoint: `experiments/flat_100k_mixed_angle30_mask_warm/checkpoints/2026-11-03_03-25-05/2/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Eval config: `configs/env/zombie_stage1.yaml`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.04 death_rate=0.64 timeout_rate=0.32
mean_hp_lost=15.10 mean_hp_lost_on_win=9.00
mean_steps=449.00 mean_time_to_kill=208.00 mean_reward=-2.75
```

The 20-seed 15% screen did not hold: only 2/50 random-spawn wins (4%). This is
the key methodological lesson: 20 seeds cannot resolve a 10–15% effect, and
three wins out of twenty is consistent with a 4–10% true rate.

The model is better at ±30° (34%) and mixed distance (48%) but does not
generalize to the full 360° angle distribution.
