# Zero-shot VPT 1x baseline

Command:

```bash
python scripts/eval_baseline.py --policy vpt --episodes 50 --seed-start 0 \
  --max-steps 600 --deterministic
```

Result:

```text
win_rate=0.02 death_rate=0.28 timeout_rate=0.70 stuck_rate=0.70
mean_hp_lost=11.58 mean_time_to_kill=382 mean_steps=535.7 mean_reward=-2.65
```

VPT survives longer than random and killed one zombie, but mostly wanders or
gets stuck. This is the baseline for PPO fine-tuning.
