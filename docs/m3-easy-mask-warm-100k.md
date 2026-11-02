# M3 — 100k Easy-AI Mask Warm Start

Date: 2026-11-03  
Source: `experiments/flat_100k_mask_warm/checkpoints/2026-11-02_23-43-23/2/`  
Config: `configs/env/zombie_stage1_easy_mask.yaml`  
Checkpoint: `experiments/flat_100k_easy_mask_warm/checkpoints/2026-11-03_00-33-34/2/`

## Result

### Easy AI front spawn (50 seeds)

```text
win_rate=0.30 death_rate=0.58 timeout_rate=0.12 stuck_rate=0.08
mean_hp_lost=14.36 mean_steps=287.34
mean_reward=+1.10 mean_time_to_kill=187.87
```

### Random spawn (20 seeds)

```text
win_rate=0.00 death_rate=0.65 timeout_rate=0.35
mean_hp_lost=15.45 mean_reward=-3.30 damage_ratio=0.174
```

## Interpretation

Compared with the previous easy-AI checkpoint:

| Metric | Previous | This run |
|---|---:|---:|
| Win rate | 0.30 | 0.30 |
| Death rate | 0.50 | 0.58 |
| Time-to-kill | 199.4 | 187.9 |
| Mean reward | +0.78 | +1.10 |

The policy kills faster but dies more. The easy-AI curriculum has plateaued at
30% wins. The bottleneck is now **defense**, not finding or hitting the target.

Random spawn is still 0%. Do not restore random spawn yet.

## Next: defense probes

Run short 10k warm-started probes from this checkpoint, one variable at a time:

1. Increase `damage_taken` from 0.2 to 0.3 (value survival more).
2. Move the easy spawn from 3–4 to 5–7 blocks (more reaction time).
3. Combine both only if the individual probes help.

Evaluate each on the same easy-AI 20 seeds and watch:

- death rate
- HP lost
- win rate
- time-to-kill

Promote the best to 100k only if it reduces deaths without collapsing the win
rate.

## Artifacts

- `experiments/20261103_010333_eval_vpt_ckpt/` (easy AI, 50 seeds)
- `experiments/20261103_010700_eval_vpt_ckpt/` (random spawn, 20 seeds)
