# HH2 High-Level PPO — RTX 4090D GPU Validation

Date: 2026-11-05
Training commit: `6094156f7f87eb74848de4b9c05fc13cc7ad9982`
Hardware: NVIDIA GeForce RTX 4090 D, 24,564 MiB VRAM
Scope: L1 engineering smoke and L2 hierarchical data-flow validation

## Result

HH2 successfully loaded the HH1 checkpoint, froze the complete low-level
Conditioned VPT, collected a real Minecraft rollout, updated the high-level
Actor-Critic with PPO, and saved a reloadable checkpoint.

L1 and the tested L2 data-flow properties passed. This is not an L3 claim that
the high-level policy has learned effective tactical behavior.

## HH1 input checkpoint

```text
experiments/h1_tactical_10k_4090d/checkpoints/2026-11-05_07-53-46/20/model.ckpt
```

SHA256:

```text
02d397d2621527c3ef479293ff728982216382c27c32b361cad917264993490f
```

The checkpoint loaded with `strict=True`.

## Harness construction

| Check | Result |
|---|---:|
| Frozen low-level parameters | 72,049,278 |
| Trainable high-level parameters | 41,987 |
| High-level feature dimension | 192 |
| Low-level eval mode | yes |
| Low-level gradients disabled | yes |
| Construction-time CUDA allocation | 275.3 MiB |

## Real GPU smoke

```text
environment steps: 100
episodes: 1
high-level transitions: 5
PPO updates: 1
peak GPU memory: 758 MiB
```

Training metrics:

| Metric | Value |
|---|---:|
| Mean episode reward | -0.1000000 |
| Total loss | -0.1707926 |
| Policy loss | -0.1662801 |
| Value loss | 0.0045900 |
| Entropy | 0.6807505 |
| Gradient norm | 0.8353970 |

All saved policy tensors were finite. The optimizer state reloaded with a
maximum step counter of 4.

## Hierarchical tactic trace

A separate 45-step real-environment trace instrumented both the public
low-level call and the conditioner's ID resolver.

| Environment step | High-level tactic |
|---:|---:|
| 1 | 0 |
| 21 | 1 |
| 41 | 1 |

Across all 45 primitive steps:

- the gate retained or refreshed the tactic at the expected interval;
- the low-level policy received the current high-level tactic;
- the conditioner resolved the same ID;
- both tactic IDs passed through the real inference chain.

## HH2 checkpoint

| File | Repository-local path | SHA256 |
|---|---|---|
| High-level model | `experiments/h2_smoke_4090d/checkpoints/high_level_latest.pt` | `97da067379dcf04dead03595103dc4f888759120b52418ac58c7b2449eff5eb2` |
| Training history | `experiments/h2_smoke_4090d/logs/history.json` | `4de79cbf60f0af7c98f3cd975ab35243cf43434cb370b6c59e95097d239b427e` |
| Run configuration | `experiments/h2_smoke_4090d/logs/run_config.json` | sanitized repository-local copy |

Strict policy reload, finite tensor checks, optimizer-state checks, update
count, and environment-step count all passed.

## Warnings

- The headless container emitted OpenAL, Narrator, and missing-texture
  warnings. They did not interrupt simulation.
- Ray was not used by HH2.
- No CUDA OOM, NaN, Inf, or process crash occurred.

## Artifacts

- committed record: `experiments/20261105_160617_hh2_4090d_validation/`
- local checkpoint: `experiments/h2_smoke_4090d/checkpoints/`
- local sanitized logs: `experiments/h2_smoke_4090d/logs/`

Checkpoint and log directories are intentionally ignored by Git. Their
standardized, versionable evidence is stored in the timestamped record above.

## Status

- [x] L1 real environment sampling
- [x] L1 PPO update
- [x] L1 checkpoint save and strict reload
- [x] L2 low-level freeze
- [x] L2 decision interval
- [x] L2 tactic propagation to the conditioner
- [x] L2 Semi-MDP transition collection
- [ ] L3 behavioral effectiveness
