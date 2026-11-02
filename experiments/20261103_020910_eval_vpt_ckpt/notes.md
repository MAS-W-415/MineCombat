# Far checkpoint — zero-shot random spawn (20 seeds)

Checkpoint: `experiments/probe_easy_far/checkpoints/2026-11-03_01-18-07/20/`  
Eval config: `configs/env/zombie_stage1.yaml`  
Seeds: 0–19, max 600 steps, deterministic

```text
win_rate=0.10 death_rate=0.75 timeout_rate=0.15 stuck_rate=0.15
mean_hp_lost=17.00 mean_hp_lost_on_win=18.00
mean_steps=345.80 mean_time_to_kill=312.00 mean_reward=-2.34
```

First non-zero random-spawn result: 2/20 wins. Both wins ended at 18 HP lost, so
they were near-deaths rather than controlled fights. A 50-seed confirmation is
needed, but the direction is consistent with the ±30° result: angle is the
bottleneck.
