# M2 — Zero-shot Baselines

Date: 2026-10-30  
Environment: zombie stage 1, normal difficulty, no natural regeneration, night vision  
Seeds: 0–49, max 600 steps  
VPT: `CraftJarvis/MineStudio_VPT.foundation_model_1x`, deterministic sampling

## Results

| Metric | Random | Zero-shot VPT 1x |
|---|---:|---:|
| Episodes | 50 | 50 |
| Win rate | 0.00 | 0.02 |
| Death rate | 0.96 | 0.28 |
| Timeout rate | 0.04 | 0.70 |
| Stuck rate | 0.04 | 0.70 |
| Mean HP lost | 17.64 | 11.58 |
| Mean time-to-kill | — | 382 |
| Mean steps | 378.8 | 535.7 |
| Mean reward | -3.91 | -2.65 |

## Interpretation

- The random policy dies quickly: 96% death rate, 17.64 HP lost on average.
- Zero-shot VPT survives longer and kills one zombie, but spends 70% of episodes
  stuck/timeout. It does not reliably approach or attack.
- This is the expected baseline for PPO: the pretrained policy has useful
  navigation and camera control, but no combat objective.

## Limitation

These 50-seed baselines were run before the `damage_dealt` observation patch,
so their `damage_ratio` is 0. The win, death, timeout, HP-loss, and step metrics
are unaffected by the reward. New runs read `custom.damage_dealt` and report a
nonzero damage ratio. See `minecombat.envs.patches` and
`tests/test_env_patches.py`.

## Artifacts

- `experiments/20261030_170440_eval_random/`
- `experiments/20261030_171310_eval_vpt/`

Both directories contain the resolved environment config, `metrics.json`,
per-episode `episodes.csv`, and the git commit recorded at run time.
