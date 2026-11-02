# Mixed + angle curriculum 100k — random spawn (20 seeds)

Checkpoint: `experiments/flat_100k_mixed_angle30_mask_warm/checkpoints/2026-11-03_03-25-05/2/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Eval config: `configs/env/zombie_stage1.yaml`  
Seeds: 0–19, max 600 steps, deterministic

```text
win_rate=0.15 death_rate=0.60 timeout_rate=0.25
mean_hp_lost=13.80 mean_hp_lost_on_win=11.00
mean_steps=368.00 mean_time_to_kill=255.67 mean_reward=-1.30
```

20-seed screen only; superseded by the 50-seed confirmation in
`experiments/20261103_041850_eval_vpt_ckpt/` (4% win). The 15% screen was small-
sample noise and is not a valid result.
