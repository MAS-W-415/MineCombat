# M3 — 100k Inventory-Mask Warm Start

Date: 2026-11-03  
Source checkpoint: `experiments/probe_mask/checkpoints/2026-11-02_23-10-56/20/`  
Config: `configs/env/zombie_stage1_stationary_attack_mask.yaml`  
Checkpoint: `experiments/flat_100k_mask_warm/checkpoints/2026-11-02_23-43-23/2/`

## Why

Phase 1 selected inventory masking as the only probe with repeatable kills:

- 20 seeds: 2 kills, 10% win, 0 deaths.
- 50 seeds: 4 kills, 8% win, 0 deaths.

The next step was a 100k warm-start from that checkpoint.

## Stationary task (50 seeds)

```text
win_rate=0.22 death_rate=0.00 timeout_rate=0.78 stuck_rate=0.34
mean_hp_lost=0.00 mean_steps=515.46
mean_reward=+2.50 mean_time_to_kill=215.73
```

22% win rate, zero deaths, positive reward. This is the first stable combat
signal in the project.

## Transfer: easy AI zombie (front, counterattacks)

| Seeds | Win rate | Death | Timeout | HP lost | Time-to-kill |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.35 | 0.50 | 0.15 | 13.90 | 221.14 |
| 50 | **0.30** | 0.50 | 0.20 | 14.56 | 199.40 |

The policy transfers to a counterattacking zombie when it spawns in front:
30% win rate. It still takes heavy damage (50% death, 14.6 HP lost), so defense
is weak.

## Transfer: random spawn (20 seeds)

```text
win_rate=0.00 death_rate=0.50 timeout_rate=0.50
mean_hp_lost=14.15 mean_reward=-3.26 damage_ratio=0.043
```

Random spawn still fails. The policy can fight a visible front target but does
not reliably find or aim at a random one.

## Conclusion

The curriculum is working:

```text
stationary NoAI: 8% -> 22% win
easy AI front: 30% win, but 50% death
random spawn: 0% win
```

Next step: warm-start 100k on the easy-AI + inventory-mask curriculum, then
reduce the death rate before restoring random spawn.

## Artifacts

- `experiments/20261103_001737_eval_vpt_ckpt/` (stationary, 50 seeds)
- `experiments/20261103_002040_eval_vpt_ckpt/` (easy AI, 20 seeds)
- `experiments/20261103_002408_eval_vpt_ckpt/` (random spawn, 20 seeds)
- `experiments/20261103_003030_eval_vpt_ckpt/` (easy AI, 50 seeds)
