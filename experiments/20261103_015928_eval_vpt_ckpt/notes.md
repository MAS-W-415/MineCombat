# Defense probe — far spawn, evaluated on far (50 seeds)

Checkpoint: `experiments/probe_easy_far/checkpoints/2026-11-03_01-18-07/20/`  
Training config: `configs/env/zombie_stage1_easy_far_mask.yaml` (5–7 blocks)  
Eval config: `configs/env/zombie_stage1_easy_far_mask.yaml` (5–7 blocks)  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.32 death_rate=0.54 timeout_rate=0.14 stuck_rate=0.12
mean_hp_lost=14.12 mean_hp_lost_on_win=7.69
mean_steps=288.74 mean_time_to_kill=215.31 mean_reward=+0.91
```

The 20-seed 50% win rate did not hold at 50 seeds: the far task itself is only
32% win with 54% deaths. The far-trained checkpoint is better at close range
than at the distance it trained on. This is why the 20-seed screen was not used
as a final result.
