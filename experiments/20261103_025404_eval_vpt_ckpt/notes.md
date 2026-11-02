# Angle curriculum 100k — random spawn (20 seeds)

Checkpoint: `experiments/flat_100k_angle30_mask_warm/checkpoints/2026-11-03_02-11-52/2/`  
Training config: `configs/env/zombie_stage1_easy_angle30_mask.yaml`  
Eval config: `configs/env/zombie_stage1.yaml`  
Seeds: 0–19, max 600 steps, deterministic

```text
win_rate=0.10 death_rate=0.40 timeout_rate=0.50
mean_hp_lost=13.15 mean_hp_lost_on_win=10.50
mean_steps=444.00 mean_time_to_kill=233.00 mean_reward=-1.83
```

Win rate stayed at 10%, but deaths fell from 75% to 40% and timeouts rose to
50%. In the 10 timeout episodes the policy dealt zero damage, so this is
avoidance/survival rather than improved engagement.
