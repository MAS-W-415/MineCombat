# 10k PPO checkpoint evaluation

Command:

```bash
python scripts/eval_baseline.py \
  --policy vpt \
  --checkpoint experiments/ppo_smoke/checkpoints/2026-10-30_18-22-45/20/model.ckpt \
  --episodes 50 --seed-start 0 --max-steps 600 --deterministic
```

Result:

```text
win_rate=0.00 death_rate=0.22 timeout_rate=0.78 stuck_rate=0.78
mean_hp_lost=10.84 mean_steps=568.22 mean_reward=-2.74
```

Comparison:

| Metric | Zero-shot VPT | 10k PPO |
|---|---:|---:|
| Win rate | 0.02 | 0.00 |
| Death rate | 0.28 | 0.22 |
| Timeout rate | 0.70 | 0.78 |
| Mean HP lost | 11.58 | 10.84 |
| Mean steps | 535.7 | 568.2 |
| Mean reward | -2.65 | -2.74 |

The 10k checkpoint survives slightly better (lower death rate and HP loss) but
does not win. This is expected: 10k steps is a pipeline smoke, not a training
run. It does not show that PPO improves over zero-shot; that question needs the
1M-step run.
