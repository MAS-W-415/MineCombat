# H1 Tactical Conditioning — GPU Validation

Date: 2026-11-02  
Commit: `7d61ba0`  
Hardware: RTX 4060 Laptop 8 GB, 31 GB system RAM, JDK 8  
Scope: engineering and conditioning validation, not a performance claim

## Result

The H1 training chain works on the real Minecraft / Ray / PPO stack:

- Conditioned VPT loads and trains.
- `tactic_ids` reach the model through rollout and PPO batches.
- The tactic reward wrapper writes `reward_base`, `reward_tactical_aux`, and
  `reward_training` into `info`.
- Checkpoints save with the conditioner parameters.
- Both tactic IDs are sampled across episodes.

Behavioral validation is **not** complete: there is no tactic-aware evaluation
script yet, so no 50-seed aggressive-vs-defensive comparison exists.

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

## What passed

- [x] Conditioned VPT training chain
- [x] Tactic IDs in rollout and PPO batches
- [x] Tactic-specific reward wrapper
- [x] Both tactics sampled
- [x] Conditioner parameters trained and finite
- [x] Action distribution changes with tactic ID
- [x] Checkpoint save / load

## What remains

- [ ] Tactic-aware evaluation script that loads `ConditionedVPTPolicy` with a
      fixed `tactic_id`.
- [ ] 50-seed aggressive vs defensive comparison.
- [ ] Video review for repeatable behavioral differences.
- [ ] Reward-hacking check for the auxiliary terms.
- [ ] Decision on whether H1 is strong enough to start H2.

## Notes

- The `/tmp/MineStudio` engine was cleaned by the system between sessions and
  had to be re-downloaded and patched before this validation. This is an
  environment issue, not a code issue.
- The `too high kl` warnings are logged but do not stop training. Monitor them
  in longer runs; a single 10k smoke is not enough to judge stability.
- The H1 runbook's behavioral section requires evaluation support that does not
  exist yet. Do not claim behavioral validation from this report.
