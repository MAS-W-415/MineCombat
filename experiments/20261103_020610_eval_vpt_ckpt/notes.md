# Far checkpoint — zero-shot mixed distance 3–7 (20 seeds)

Checkpoint: `experiments/probe_easy_far/checkpoints/2026-11-03_01-18-07/20/`  
Eval config: `configs/env/zombie_stage1_easy_mixed_mask.yaml`  
Seeds: 0–19, max 600 steps, deterministic

```text
win_rate=0.40 death_rate=0.50 timeout_rate=0.10 stuck_rate=0.10
mean_hp_lost=11.40 mean_hp_lost_on_win=3.75
mean_steps=243.80 mean_time_to_kill=185.88 mean_reward=+2.28
```

Distance is not the main bottleneck: the far-trained checkpoint already holds
40% wins when distance is mixed between 3 and 7 blocks, and wins at close range
take only 3.8 HP of damage.
