# Far checkpoint — zero-shot ±30° spawn jitter (20 seeds)

Checkpoint: `experiments/probe_easy_far/checkpoints/2026-11-03_01-18-07/20/`  
Eval config: `configs/env/zombie_stage1_easy_angle30_mask.yaml`  
Seeds: 0–19, max 600 steps, deterministic

```text
win_rate=0.10 death_rate=0.80 timeout_rate=0.10 stuck_rate=0.05
mean_hp_lost=17.70 mean_hp_lost_on_win=15.00
mean_steps=324.95 mean_time_to_kill=365.00 mean_reward=-2.32
```

A small ±30° spawn offset collapses the policy: 10% win and 80% deaths, down
from 52% / 40% at the same distance with a fixed front spawn. Angle handling is
the next bottleneck, so the next 100k run trains on this config.
