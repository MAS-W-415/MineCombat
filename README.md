# MineCombat

**Curriculum reinforcement learning for Minecraft combat — from zombies to the Ender Dragon.**

MineCombat is a research project built on [MineStudio](https://github.com/CraftJarvis/MineStudio) and [VPT](https://github.com/openai/Video-Pre-Training). The agent controls the native keyboard/mouse action space — camera, movement, attack, jump, sprint — from 128×128 RGB frames. The long-term goal is an end-to-end policy that can fight the Ender Dragon; the first milestone is a reproducible zombie arena.

[中文说明](README.zh-CN.md)

## Status

`M0 — environment bring-up`, `M1 — zombie arena`, and `M2 — baselines` are
complete. `M3 — first PPO` is in progress: the single-GPU pipeline, 10k smoke,
dense damage signal, and 50-seed checkpoint evaluation work. The 1M-step run is
blocked on a 24 GB GPU / 64 GB RAM host; the local RTX 4060 runs one environment
at ~45 steps/s.

Key results:

- [M2 baselines](docs/m2-baseline.md): random 0% win / 96% death; zero-shot VPT 2% win / 28% death.
- [M3 PPO smoke](docs/m3-ppo-smoke.md): 10k steps, 20 updates, 6.3 GiB peak GPU memory.
- [Dense damage](docs/m3-dense-damage.md): first nonzero damage ratio (0.0897) at 10k.
- [1M runbook](docs/1m-run.md): exact command and preflight for the next host.

The first baseline and PPO runs are tracked in [Issues](https://github.com/MAS-W-415/MineCombat/issues).

## Why this project

Most Minecraft agents are evaluated on exploration or item-collection tasks. Combat is a useful next step because it requires:

- visual target detection and camera control,
- approach and spacing decisions,
- timing of attacks and retreats,
- risk management under partial observability,
- transfer from one enemy type to another.

The project studies a curriculum from a single zombie to mixed enemies and, eventually, the Ender Dragon. Each stage is a self-contained RL experiment with measurable win rate, damage taken, and time-to-kill.

## Curriculum

| Stage | Enemy | Skills | Status |
|---|---|---|---|
| 0 | Environment | Stable reset, rendering, VPT inference, logging | Complete |
| 1 | Zombie | Approach, aim, attack, avoid damage | Arena + baselines complete; PPO in progress |
| 2 | Skeleton | Projectile avoidance, distance control | Planned |
| 3 | Creeper + mixed | Explosion risk, multiple enemies, terrain | Planned |
| 4 | Ender Dragon | Long-horizon tactics, high-level harness | Planned |

## Method

- **Policy:** VPT 1x foundation model, fine-tuned with PPO.
- **Observation:** 128×128 RGB plus the previous action; HP is logged for reward and evaluation but is not a policy input in stage 1.
- **Action space:** MineStudio `agent` action space (native camera and button controls).
- **Reward:** dense combat reward combining damage dealt, damage taken, kill bonus, death penalty, and a small time penalty.
- **Evaluation:** fixed-seed episodes with win rate, HP lost, time-to-kill, damage ratio, and stuck rate.

The initial reward is:

```text
r_t = 0.01 * Δdamage_dealt   # damage_dealt is in tenths of a HP
    - 0.2 * Δhp_lost
    - 0.001
    + 10 * kill
    - 5 * death
```

Weights are starting points and will be ablated. See [docs/rewards.md](docs/rewards.md).

## Repository layout

```text
configs/        Environment, PPO, and evaluation configurations
src/minecombat/ Environment wrappers, callbacks, training and evaluation code
scripts/        Setup, smoke test, baseline evaluation, and training entry points
docs/           Research plan, environment notes, reward design, roadmap
experiments/    Run logs and reproducibility notes (large artifacts are ignored)
tests/          Fast unit tests for reward math, metrics, and configs
```

## Quickstart

The project requires Python 3.10, JDK 8, and a working OpenGL renderer. See [docs/environment.md](docs/environment.md).

```bash
git clone git@github.com:MAS-W-415/MineCombat.git
cd MineCombat
bash scripts/setup.sh
source .venv/bin/activate

# Random-action smoke test
python scripts/smoke_test.py --config configs/env/zombie_stage1.yaml --steps 100

# Baseline evaluation
python scripts/eval_baseline.py --policy random --episodes 5
```

For VPT inference, download the 1x checkpoint first:

```bash
python - <<'PY'
from minestudio.models import VPTPolicy
VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.foundation_model_1x")
PY
```

Then run:

```bash
python scripts/smoke_test.py --policy vpt --steps 600
```

## Training

The local RTX 4060 Laptop has 8 GB VRAM, so the smoke profile uses one
environment. The base config is sized for a 24 GB GPU with 8 environments,
batch size 4, context length 64, and VPT 1x.

```bash
# Local pipeline smoke (one environment)
bash scripts/start_ray.sh
bash scripts/train_zombie.sh --smoke

# 1M-step run on a 24 GB GPU / 64 GB RAM host
bash scripts/train_zombie.sh --num-iterations 20 --num-envs 8 --batch-size 4
```

The official MineStudio configuration assumes multiple GPUs. Use
`configs/ppo/zombie_vpt1x.yaml` for the single-GPU profile and see
[docs/1m-run.md](docs/1m-run.md) for the full runbook.

## Research questions

- **H1:** PPO from VPT 1x improves zombie win rate and reduces damage taken versus zero-shot VPT.
- **H2:** A zombie-pretrained policy transfers to skeletons more sample-efficiently than direct VPT fine-tuning.
- **H3:** A high-level harness improves long-horizon boss performance once the flat policy is strong.

The flat VPT + PPO policy is the control condition. The harness is intentionally deferred until the zombie stage is stable, so that any gain can be attributed to the hierarchy rather than to a stronger base policy.

## Roadmap

| Milestone | Goal | Acceptance |
|---|---|---|
| M0 | Environment bring-up | Stable simulator, VPT 1x inference, FPS/VRAM recorded |
| M1 | Zombie arena | 20 reproducible resets, random spawn and yaw |
| M2 | Baseline and metrics | 50-seed zero-shot evaluation, metrics report |
| M3 | First PPO run | 1M steps, learning curve, stage-1 gate |
| M4 | Curriculum | Skeleton extension only after M3 passes |

See [docs/roadmap.md](docs/roadmap.md) for details.

## Acknowledgements

This project builds on:

- [MineStudio](https://github.com/CraftJarvis/MineStudio) for the simulator, data, models, and online RL infrastructure.
- [MineRL](https://github.com/minerllabs/minerl) and [Project Malmo](https://github.com/microsoft/malmo) for the underlying Minecraft environment.
- [OpenAI VPT](https://github.com/openai/Video-Pre-Training) for the pretrained foundation model.

## License

MIT. See [LICENSE](LICENSE).
