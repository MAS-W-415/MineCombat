# M0 — Environment Bring-up Report

Date: 2026-10-30  
Machine: RTX 4060 Laptop, 8 GB VRAM, 31 GB RAM, 24 CPU threads

## Result

M0 is complete on the local machine. MineStudio, the Minecraft simulator, and
the VPT 1x policy all run. The first zombie arena reset and a 600-step VPT
episode completed without a crash.

## Measurements

| Metric | Value |
|---|---|
| Random-policy throughput | ~102 steps/s |
| VPT 1x throughput | ~67–78 steps/s |
| Peak VRAM (simulator + VPT) | ~1078 MiB |
| Fast reset time | ~3.7–4.1 s |
| 600-step VPT episode | completed |
| VPT model VRAM | ~0.27 GB |
| Engine download | ~458 MB |

## What was verified

- MineStudio 1.1.6 imports and launches the simulator.
- 128×128 RGB observations and the expected `info` keys are present.
- A zombie spawns after reset and approaches the player.
- Player health loss is reflected in `info["health"]`.
- The combat reward penalizes HP loss.
- `JudgeResetCallback` terminates the episode at 600 steps.
- VPT 1x loads from Hugging Face and produces actions at interactive speed.
- A 300-step VPT video was recorded locally at
  `experiments/m0_vpt_baseline/vpt_300_steps.mp4`.

## Environment issues found

1. **No JDK 8 in the system.** Installed a user-local OpenJDK 8 from Ubuntu
   packages under `~/.local/opt/jdk8`.
2. **`python3-venv` is missing.** Used `virtualenv` from user site-packages.
3. **MineStudio does not declare `psutil`.** Added it to project dependencies.
4. **Hugging Face is unreachable directly.** Used
   `HF_ENDPOINT=https://hf-mirror.com`.
5. **`xvfb-run` is missing** even though `DISPLAY=:0` works. Added a local shim
   that runs Java on the existing display.
6. **MalmoEnv startup race.** MineStudio waits for `CLIENT DORMANT` but not for
   the Malmo TCP server. Added a 3-second grace period via
   `scripts/patch_minestudio.py`.
7. **Stale instance state after a crashed run.** Clearing
   `/tmp/MineStudio/MP_tmp` and `/tmp/MineStudio/tmp` fixes repeated
   `Connection refused` errors.
8. **Teleport rotation syntax is fragile.** `/tp ... 0 0` can be parsed as
   something other than rotation; use `0.0` for the pitch component.

## Verified in later milestones

- A successful player attack and zombie kill: M1, after the engine
  `leftClickCounter` patch.
- `damage_dealt` and `kill_entity` increments: M1; the `custom.damage_dealt`
  observation was fixed in M3.
- PPO training and VRAM usage: M3 smoke, see `docs/m3-ppo-smoke.md`.
- Reproducibility across 20 resets: M1, see `scripts/validate_resets.py`.

## Reproduction

```bash
export JAVA_HOME="$HOME/.local/opt/jdk8/usr/lib/jvm/java-8-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
source .venv/bin/activate

bash scripts/clean_minestudio_runtime.sh
python scripts/smoke_test.py --config configs/env/zombie_stage1.yaml --steps 100
python scripts/smoke_test.py --policy vpt --steps 600
```
