# HH1 Tactical Conditioning — GPU Validation

Date: 2026-11-02  
Training commit: `7d61ba0`  
Evaluation commit: `2e858eb` (includes `aa5d140`, the conditioned eval CLI)  
Hardware: RTX 4060 Laptop 8 GB, 31 GB system RAM, JDK 8  
Scope: engineering and conditioning validation, not a performance claim

## Result

The HH1 training chain works on the real Minecraft / Ray / PPO stack:

- Conditioned VPT loads and trains.
- `tactic_ids` reach the model through rollout and PPO batches.
- The tactic reward wrapper writes `reward_base`, `reward_tactical_aux`, and
  `reward_training` into `info`.
- Checkpoints save with the conditioner parameters.
- Both tactic IDs are sampled across episodes.

A 50-seed aggressive-vs-defensive comparison is now included. It shows a small
but directionally consistent behavioral difference; it is not a win-rate
improvement. Video review and reward-hacking analysis are still pending.

## 512-step smoke

```bash
bash scripts/train_zombie.sh \
  --conditioned --num-tactics 2 \
  --sample-tactic-per-episode --tactical-rewards \
  --smoke \
  --save-path experiments/h1_tactical_smoke \
  --name h1_tactical_smoke
```

Result:

```text
512 environment steps
1 PPO update
89.0 s wall time
checkpoint: experiments/h1_tactical_smoke/checkpoints/2026-11-02_14-04-25/1
```

The checkpoint contains `conditioner.embedding.weight` `(2, 1024)` and
`conditioner.projection.weight` `(1024, 1024)`; all tensors are finite.

## 10k diagnostic

```bash
bash scripts/train_zombie.sh \
  --conditioned --num-tactics 2 \
  --sample-tactic-per-episode --tactical-rewards \
  --smoke --num-iterations 20 \
  --save-path experiments/h1_tactical_10k \
  --name h1_tactical_10k
```

Result:

```text
10,240 environment steps
20 PPO updates
236.2 s wall time
checkpoint: experiments/h1_tactical_10k/checkpoints/2026-11-02_14-08-52/20
```

Training emitted 4 `too high kl` warnings but continued. The final checkpoint
has no NaN or non-finite tensors.

## Conditioner training

Comparing the 1-update smoke checkpoint with the 20-update 10k checkpoint:

| Parameter | Delta norm | Note |
|---|---:|---|
| `conditioner.embedding.weight` | 1.299 | tactic embeddings updated |
| `conditioner.projection.weight` | 0.162 | zero-init projection moved |

The projection norm grew from 0.026 to 0.159, so gradients reach the
conditioning module.

## Tactic sampling and reward decomposition

Ten short episodes (40 steps each) with `sample_per_episode=True`:

```text
tactic 0: 6 episodes
tactic 1: 4 episodes
```

Each step exposed `reward_base`, `reward_tactical_aux`, and `reward_training`.
With a random policy and no combat, the auxiliary reward was correctly zero.

Scripted real-environment checks:

```text
aggressive: base=+11.937 aux=+0.995 training=+12.932
defensive:  base=-3.722  aux=-1.800 training=-5.522
```

So tactic 0 rewards damage and tactic 1 penalizes HP loss as designed.

## Action-distribution sensitivity

Loaded the 10k checkpoint and compared the same observation under tactic 0 and
tactic 1, with identical recurrent state:

| Metric | Mean |
|---|---:|
| Button logit KL | 5.77e-4 |
| Camera logit KL | 3.51e-4 |
| Max button logit difference | ~0.082 |
| Max camera logit difference | ~0.069 |
| Max value difference | 9.4e-5 |

The effect is nonzero, so conditioning is not ignored. It is small after 10k
steps, which is expected given the zero-initialized projection. This is **not**
evidence that the tactics produce useful behavior.

Reproduce with:

```bash
python scripts/diagnose_h1.py \
  --checkpoint experiments/h1_tactical_10k/checkpoints/2026-11-02_14-08-52/20/model.ckpt
```

The script exits nonzero if the conditioning is ignored or any metric is
non-finite.

## 50-seed behavioral evaluation

Command:

```bash
python scripts/eval_conditioned.py \
  --checkpoint experiments/h1_tactical_10k/checkpoints/2026-11-02_14-08-52/20/model.ckpt \
  --num-tactics 2 --tactic-ids 0 1 \
  --episodes 50 --seed-start 0 --max-steps 600 --deterministic
```

| Metric | Aggressive (0) | Defensive (1) |
|---|---:|---:|
| Win rate | 0.00 | 0.00 |
| Death rate | 0.36 | 0.32 |
| Timeout rate | 0.64 | 0.68 |
| Stuck rate | 0.60 | 0.66 |
| Mean HP lost | 12.04 | 11.24 |
| Mean steps | 536.26 | 534.62 |
| Mean reward | -2.87 | -2.72 |
| Damage ratio | 0.0651 | 0.0548 |

Aggressive deals more damage; defensive takes less damage and dies less often.
The direction matches the intended tactics, but the effect is small and both
conditions still have a 0% win rate after 10k steps. This is evidence of
conditioning sensitivity, not evidence of tactical success.

Artifacts:

- `experiments/20261102_143711_eval_conditioned_tactic_0/`
- `experiments/20261102_144410_eval_conditioned_tactic_1/`

## What passed

- [x] Conditioned VPT training chain
- [x] Tactic IDs in rollout and PPO batches
- [x] Tactic-specific reward wrapper
- [x] Both tactics sampled
- [x] Conditioner parameters trained and finite
- [x] Action distribution changes with tactic ID
- [x] Checkpoint save / load
- [x] Tactic-aware evaluation CLI
- [x] 50-seed aggressive vs defensive comparison

## What remains

- [ ] Video review for repeatable behavioral differences.
- [ ] Reward-hacking check for the auxiliary terms.
- [ ] Decision on whether HH1 is strong enough to start HH2.
- [ ] Longer training before claiming tactical behavior; 10k is a diagnostic.

## Notes

- The `/tmp/MineStudio` engine was cleaned by the system between sessions and
  had to be re-downloaded and patched before this validation. This is an
  environment issue, not a code issue.
- The `too high kl` warnings are logged but do not stop training. Monitor them
  in longer runs; a single 10k smoke is not enough to judge stability.
- The 50-seed comparison uses a 10k checkpoint. The differences are small and
  the win rate is still zero; do not describe this as a successful tactical
  policy or as HH2 readiness.
