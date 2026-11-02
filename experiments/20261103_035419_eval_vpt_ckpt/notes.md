# Mixed + angle curriculum 100k — easy front (50 seeds)

Checkpoint: `experiments/flat_100k_mixed_angle30_mask_warm/checkpoints/2026-11-03_03-25-05/2/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Eval config: `configs/env/zombie_stage1_easy.yaml`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.44 death_rate=0.48 timeout_rate=0.08
mean_hp_lost=12.38 mean_hp_lost_on_win=6.68
mean_steps=272.00 mean_time_to_kill=192.14 mean_reward=+2.63
```

Close-range retention is 44%, down from the far checkpoint's 52% but well above
the 40% keep threshold. Deaths rose from 40% to 48%, a mild regression. HP lost
per win improved from 6.81 to 6.68.
