# M3 — PPO Pipeline Smoke Test

Date: 2026-10-30  
Hardware: RTX 4060 Laptop, 8 GB VRAM

## Result

The single-GPU PPO pipeline runs end to end:

- Ray head with `database`, `rollout_workers`, and `wandb` resources.
- Detached `RolloutManager` and `ReplayBufferActor`.
- One rollout worker with one Minecraft environment.
- Ray Train worker with VPT 1x on `cuda`.
- Two fragments prepared, PPO update 1 completed.
- Checkpoint saved with `model.ckpt` and `optimizer.ckpt`.
- Environment worker reached 600 steps; episode statistics were reported.

Command:

```bash
bash scripts/start_ray.sh
bash scripts/train_zombie.sh --smoke
```

The smoke profile is:

- `num_iterations: 1`
- `batches_per_iteration: 2`
- `num_envs: 1`
- `fragment_length: 256`
- `context_length: 64`

## 10k-step smoke

Command:

```bash
bash scripts/start_ray.sh
bash scripts/train_zombie.sh --smoke --num-iterations 20
```

Result:

- 20 PPO iterations completed.
- `20 × 2 fragments × 256 steps ≈ 10,240` environment steps.
- Checkpoint saved at `experiments/ppo_smoke/checkpoints/2026-10-30_18-22-45/20`.
- No OOM, no trainer crash.
- Runtime: 226 s from `Start training` to the final checkpoint
  (≈45 environment steps/s with one environment).
- Final episode statistics: mean reward ≈ -2.83, episode length ≈ 566.

### VRAM

A separate 2-iteration run was monitored with `nvidia-smi` once per second:

```text
peak total GPU memory: 6484 MiB (~6.3 GiB)
```

This includes about 0.6 GiB of desktop overhead, so the training stack itself
uses roughly 5.9 GiB. It fits in the 8 GB budget, but leaves little headroom for
a second environment on this GPU.

## Issues found

1. **Ray resources** — MineStudio requests custom resources `database`,
   `rollout_workers`, and `wandb`. A plain `ray.init()` cannot schedule the
   replay buffer or rollout worker.
2. **Ray namespace** — the rollout manager and trainer must share
   `namespace="online"`.
3. **JDK 8 in Ray workers** — starting Ray from a shell with Java 11 causes
   `javax/xml/bind/JAXBException`. `scripts/start_ray.sh` injects the local
   JDK 8.
4. **Multiprocessing pickle** — `env_generator` and `policy_generator` must live
   in an importable module (`minecombat.train.generators`), not in `__main__`.
5. **Multi-env capacity** — two clients share `DISPLAY=:0` and work, but four
   clients exhaust the 31 GB system RAM (29.55 GB used) even though VRAM stays
   low. Two environments also halve throughput (~22 vs ~45 steps/s) because
   the clients compete for CPU and the single GPU. The local smoke therefore
   uses one environment; 8-environment training needs more system RAM and a
   larger GPU.
6. **Value-head warning** — MineStudio's `NormalizeEwma` turns a `(1, 64)`
   value target into `(1, 1, 64)`, so `ScaledMSEHead` warned about broadcasting.
   The loss was numerically unchanged, but the warning hid real shape errors.
   `minecombat.train.patches.patch_scaled_mse_head()` now reshapes the
   normalized target back to the prediction shape when the element count
   matches. The warning is gone in the 1-iteration smoke, and
   `tests/test_train_patches.py` checks the loss is identical.

## Next

- Start the 1M-step run on a host with more system RAM and a 4090/cloud GPU.
  Do not run it on the local 4060 with one environment: at ~45 steps/s it
  would take about 6 hours, and four local environments already exhaust the
  31 GB system RAM.
