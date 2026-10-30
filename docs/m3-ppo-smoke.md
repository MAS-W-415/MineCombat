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
5. **Single display** — the local `xvfb-run` shim shares `DISPLAY=:0`; multiple
   Minecraft clients conflict, so the smoke uses one environment.
6. **Value-head warning** — MineStudio's `ScaledMSEHead` warns that the value
   target shape `(1, 1, 64)` differs from the prediction `(1, 64)`. The update
   runs, but this should be reviewed before the 1M-step run.

## Next

- Install real Xvfb or VirtualGL for multiple environments.
- Review the value-head shape warning.
- Start the 1M-step run on a 4090 or cloud GPU. Do not run it on the local
  4060 with one environment: at ~45 steps/s it would take about 6 hours, and
  the single display prevents parallel environments.
