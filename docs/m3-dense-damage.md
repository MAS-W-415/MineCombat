# M3 — Dense Damage Signal

Date: 2026-10-30

## Root cause

The engine does emit `damage_dealt`, but under `info["custom"]["damage_dealt"]`.
MineStudio 1.1.6's `ObserveFromFullStats.from_hero` only looked at the top level
of the info dict, so the value was silently dropped and the dense damage reward
stayed at zero.

This was not an engine bug. A scripted zombie kill shows the raw stat:

```text
k 20  custom {'damage_dealt': 98}
k 40  custom {'damage_dealt': 196}
k 52  custom {'damage_dealt': 199}  kill_entity={'zombie': 1}
```

The stat is in tenths of a HP: 199 ≈ 20 HP.

## Fix

`minecombat.envs.patches.patch_observe_from_full_stats()` reads a stat from
`info["custom"]` when it is not present at the top level. `make_zombie_env`
applies the patch before creating the simulator. Top-level stats such as
`kill_entity` keep the original lookup.

The reward weight is `0.01` per stat point (0.1 per HP), so a full zombie gives
about 2.0 damage reward, below the 10.0 kill reward.

## Validation

- `tests/test_env_patches.py` checks the custom-stat lookup.
- `tests/test_reward_math.py` covers scalar and dict damage counters.
- `scripts/test_melee.py` now requires damage and kills:

```text
step=144 reward=+10.089 damage=201.0 kills=1.0 outcome=success terminated=True
RESULT damage=201.0 kills=1.0 outcome=success reward=11.865 passed=True
```

- Two consecutive episodes confirm the cumulative stat is differenced correctly
  across resets: episode 1 ends at 199, episode 2 starts at 199 and ends at 398,
  with the same total reward.

## 10k smoke with dense damage

Command:

```bash
bash scripts/train_zombie.sh --smoke --num-iterations 20 \
  --save-path experiments/ppo_10k_damage
```

Result:

```text
10,240 environment steps
238.8 s
checkpoint: experiments/ppo_10k_damage/checkpoints/2026-10-30_19-20-20/20
```

Evaluation on 50 seeds:

| Metric | Zero-shot VPT | 10k, no damage term | 10k, dense damage |
|---|---:|---:|---:|
| Win rate | 0.02 | 0.00 | 0.00 |
| Death rate | 0.28 | 0.22 | 0.36 |
| Timeout rate | 0.70 | 0.78 | 0.64 |
| Mean HP lost | 11.58 | 10.84 | 10.92 |
| Mean steps | 535.7 | 568.2 | 527.3 |
| Damage ratio | 0.00* | 0.00* | 0.0897 |

\* The first two evaluations predate the observation fix, so their damage ratio
is zero by construction.

The dense-damage policy attacks more (timeout drops from 78% to 64% and the
damage ratio becomes nonzero) but also dies more (22% to 36%) and still does not
win. Ten thousand steps is a pipeline smoke, not a training result. The 1M run
is needed to test whether the dense signal improves the win rate.

## Artifacts

- `experiments/ppo_10k_damage/checkpoints/2026-10-30_19-20-20/20/`
- `experiments/20261030_193103_eval_vpt_ckpt/`
