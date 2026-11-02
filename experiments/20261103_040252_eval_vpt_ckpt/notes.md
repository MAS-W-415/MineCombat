# Mixed + angle curriculum 100k — mixed distance (20 seeds)

Checkpoint: `experiments/flat_100k_mixed_angle30_mask_warm/checkpoints/2026-11-03_03-25-05/2/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Eval config: `configs/env/zombie_stage1_easy_mixed_mask.yaml`  
Seeds: 0–19, max 600 steps, deterministic

```text
win_rate=0.55 death_rate=0.35 timeout_rate=0.10
mean_hp_lost=11.15 mean_hp_lost_on_win=6.00
mean_steps=259.00 mean_time_to_kill=207.91 mean_reward=+4.23
```

20-seed screen only; superseded by the 50-seed confirmation in
`experiments/20261103_041145_eval_vpt_ckpt/` (48% win). This is the best
mixed-distance result so far (previous zero-shot far checkpoint: 40%).
