# Random baseline

Command:

```bash
python scripts/eval_baseline.py --policy random --episodes 50 --seed-start 0 --max-steps 600
```

Result:

```text
win_rate=0.00 death_rate=0.96 timeout_rate=0.04
mean_hp_lost=17.64 mean_steps=378.8 mean_reward=-3.91
```

The random policy dies quickly and never kills the zombie. This is the lower
bound for stage 1.
