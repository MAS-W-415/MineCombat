# 10k PPO with dense damage — checkpoint evaluation

Checkpoint: `experiments/ppo_10k_damage/checkpoints/2026-10-30_19-20-20/20/model.ckpt`  
Seeds: 0–49, max 600 steps, deterministic

Result:

```text
win_rate=0.00 death_rate=0.36 timeout_rate=0.64 stuck_rate=0.62
mean_hp_lost=10.92 mean_steps=527.28 mean_reward=-2.70
damage_ratio=0.0897
```

This is the first run with a nonzero damage ratio. The policy attacks more often
(timeout down from 78% to 64%) but also dies more (22% to 36%) and still never
wins. Ten thousand steps is a pipeline smoke, not a training result.

Comparison:

| Metric | Zero-shot VPT | 10k, no damage term | 10k, dense damage |
|---|---:|---:|---:|
| Win rate | 0.02 | 0.00 | 0.00 |
| Death rate | 0.28 | 0.22 | 0.36 |
| Timeout rate | 0.70 | 0.78 | 0.64 |
| Mean HP lost | 11.58 | 10.84 | 10.92 |
| Mean steps | 535.7 | 568.2 | 527.3 |
| Damage ratio | 0.00* | 0.00* | 0.0897 |

\* The first two evaluations predate the `custom.damage_dealt` observation fix,
so their damage ratio is zero by construction, not because the policies never
dealt damage.
