# MineCombat

**从僵尸到末影龙：Minecraft 战斗的课程强化学习。**

MineCombat 是一个基于 [MineStudio](https://github.com/CraftJarvis/MineStudio) 和 [VPT](https://github.com/openai/Video-Pre-Training) 的研究项目。智能体直接从 128×128 RGB 画面学习控制原生键鼠动作——视角、移动、攻击、跳跃、冲刺。长期目标是端到端击败末影龙，第一个里程碑是可复现的僵尸战斗场景。

[English README](README.md)

## 当前状态

`M0 — 环境打通`、`M1 — 僵尸竞技场`、`M2 — 基线与指标` 已完成。
`M3 — 首次 PPO` 进行中：1M flat PPO checkpoint 在 easy 正面 72%、±30° 52%、
组合距离+角度 40.7%、全随机 22.7%（均为 3×50 局）。HH1/HH2 已在 RTX 4090D
上完成工程验证；行为实验仍等 flat Stage-1 gate。

主要结果：

- [M2 基线](docs/m2-baseline.md)：随机策略 0% 胜率 / 96% 死亡率；zero-shot VPT 2% 胜率 / 28% 死亡率。
- [M3 PPO smoke](docs/m3-ppo-smoke.md)：10k steps、20 次 update、峰值显存 6.3 GiB。
- [Dense damage](docs/m3-dense-damage.md)：10k 时首次出现非零伤害比（0.0090）。
- [100k flat PPO](docs/m3-100k-eval.md)：0% 胜率、16% 死亡率、伤害比 0.0313；有微弱学习信号，但还不是可用策略。
- [R1 奖励消融](docs/m3-100k-reward-r1.md)：加强进攻奖励没有效果，死亡更多、伤害更少。
- [简单正前方课程](docs/m3-100k-easy-curriculum.md)：首次出现击杀，但 50 局只有 2 局造成伤害，攻击行为仍不稳定。
- [静止靶课程](docs/m3-100k-stationary.md)：0 击杀；动作诊断显示攻击键按下率低于 1%，瓶颈是“不会发现攻击动作”。
- [静止靶 + 攻击奖励](docs/m3-100k-stationary-attack.md)：有伤害的局数从 0/50 升到 7/50，但仍 0 击杀。
- [Phase 1 探针](docs/m3-phase1-probes.md)：屏蔽背包动作后首次出现可复现击杀；静止靶 8% 胜率、0 死亡。
- [100k mask warm-start](docs/m3-mask-warm-100k.md)：静止靶 22% 胜率，正前方 AI 僵尸 30% 胜率；随机刷新仍是 0%。
- [100k easy-AI warm-start](docs/m3-easy-mask-warm-100k.md)：胜率仍 30%，但死亡率升到 58%；简单课程进入平台期，下一步做防御探针。
- [防御探针](docs/m3-defense-probes.md)：拉远刷新训练后在 easy 任务上达到 52% 胜率 / 40% 死亡；当前瓶颈是 ±30° 角度，而不是距离。
- [角度课程](docs/m3-angle-curriculum.md)：100k 的 ±30° 训练只到 16% 胜率；下一步是混合距离 + 角度的联合课程。
- [混合距离 + 角度课程](docs/m3-mixed-angle-curriculum.md)：联合 100k 在 ±30° 到 34%、混合距离到 48%；扩到 ±60° 后 easy 升到 58%，但 ±60° 目标崩到 8%。相机熵无效，问题是无方向线索的搜索；下一步需要确定 gate 定义。
- [评测方差](docs/m3-evaluation-variance.md)：同一 checkpoint、同一 seeds 在组合任务上三次跑出 10%–24%，因为僵尸 AI 跨局没有重置；以后关键结论必须 3×50 局。
- [Stage-1 重基线](docs/m3-gate-rebaseline.md)：3×50 均值组合任务 16%（10–22%）、全随机 10%（8–14%）；gate 未通过。
- [1M flat PPO](docs/m3-1m-flat-ppo.md)：easy 72%、±30° 52%、组合 40.7%（32–52%）、random 22.7%（20–26%）；属于情况 B，1M continuation 正在跑。
- [HH1/HH2 4090D 验证](docs/hh1-4090d-validation.md)：24 GB GPU 上完成工程链验证；不声称行为效果。
- [1M 运行手册](docs/1m-run.md) 和 [4090D handoff](docs/1m-4090d-handoff.md)：下一台机器上的完整命令与前置检查。

术语说明见 [docs/glossary.md](docs/glossary.md)：`M` 是里程碑，`RQ` 是研究问题，`HH` 是 Harness 阶段。统一路线见 [docs/unified-roadmap.md](docs/unified-roadmap.md)：flat PPO 是对照组，Harness 是同一路线上的扩展。公共训练预检见 [docs/training-common.md](docs/training-common.md)。

基线与 PPO 实验在 [Issues](https://github.com/MAS-W-415/MineCombat/issues) 中跟踪。

## 为什么做这个项目

现有 Minecraft 智能体多在探索或收集任务上评估。战斗是一个有价值的下一阶段，因为它同时要求：

- 视觉目标识别与视角控制
- 接近、保持距离和走位
- 攻击与后撤的时机判断
- 部分可观测下的风险管理
- 从一种敌人迁移到另一种敌人

项目采用课程学习：单个僵尸 → 骷髅 → 苦力怕与混合敌人 → 末影龙。每个阶段都是独立、可量化的 RL 实验。

## 课程路线

| 阶段 | 敌人 | 技能 | 状态 |
|---|---|---|---|
| 0 | 环境 | 稳定 reset、渲染、VPT 推理、日志 | 已完成 |
| 1 | 僵尸 | 接近、瞄准、攻击、躲避伤害 | 场景与基线已完成；PPO 进行中 |
| 2 | 骷髅 | 弹道规避、距离控制 | 计划中 |
| 3 | 苦力怕 + 混合 | 爆炸风险、多敌人、地形 | 计划中 |
| 4 | 末影龙 | 长程战术、高层 Harness | 计划中 |

## 方法

- **策略**：VPT 1x foundation model，使用 PPO 微调。
- **观测**：128×128 RGB + 上一帧动作；HP 只用于奖励和评估，第一版不作为策略输入。
- **动作空间**：MineStudio `agent` 动作空间（原生视角与按键）。
- **奖励**：伤害输出、受伤、击杀、死亡和时间惩罚组成的稠密奖励。
- **评估**：固定 seed，统计胜率、掉血、击杀时间、伤害比和卡住率。

初始奖励：

```text
r_t = 0.01 * Δdamage_dealt   # damage_dealt 以 0.1 HP 为单位
    - 0.2 * Δhp_lost
    - 0.001
    + 10 * kill
    - 5 * death
```

权重只是起点，后续会做消融。详见 [docs/rewards.md](docs/rewards.md)。

## 目录结构

```text
configs/        环境、PPO 和评估配置
src/minecombat/ 环境封装、callbacks、训练与评估代码
scripts/        安装、冒烟测试、基线评估和训练入口
docs/           研究计划、环境说明、奖励设计、路线图
experiments/    实验记录与复现说明（大文件不入库）
tests/          奖励、指标和配置的快速单元测试
```

## 快速开始

需要 Python 3.10、JDK 8 和可用的 OpenGL 渲染环境。详见 [docs/environment.md](docs/environment.md)。

```bash
git clone git@github.com:MAS-W-415/MineCombat.git
cd MineCombat
bash scripts/setup.sh
source .venv/bin/activate

# 随机动作冒烟测试
python scripts/smoke_test.py --config configs/env/zombie_stage1.yaml --steps 100

# 基线评估
python scripts/eval.py --policy random --episodes 5
```

VPT 推理需要先下载 1x checkpoint：

```bash
python - <<'PY'
from minestudio.models import VPTPolicy
VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.foundation_model_1x")
PY
```

然后：

```bash
python scripts/smoke_test.py --policy vpt --steps 600
```

## 训练

本地 RTX 4060 Laptop 只有 8 GB VRAM，所以 smoke 只开 1 个环境。
基础配置面向 24 GB GPU：8 个环境、batch size 4、context length 64、VPT 1x。

```bash
# 本地链路 smoke（单环境）
bash scripts/start_ray.sh
bash scripts/train_zombie.sh --smoke

# 24 GB GPU / 64 GB RAM 机器上的 1M 长跑
bash scripts/train_zombie.sh --num-iterations 20 --num-envs 8 --batch-size 4
```

MineStudio 官方配置默认需要多张 GPU。单卡配置见
`configs/ppo/zombie_vpt1x.yaml`；完整长跑步骤见 [docs/1m-run.md](docs/1m-run.md)。

## 研究问题

- **RQ1**：从 VPT 1x 做 PPO，能否提升僵尸胜率并降低掉血？
- **RQ2**：僵尸预训练策略迁移到骷髅，是否比直接从 VPT 微调更省样本？
- **RQ3**：在 flat policy 足够强之后，高层 Harness 是否能提升末影龙长程任务的胜率？

第一版保留 **flat VPT + PPO** 作为对照组。Harness 推迟到僵尸阶段稳定之后再加入，避免无法区分收益来源。

## 路线图

| 里程碑 | 目标 | 验收标准 |
|---|---|---|
| M0 | 环境打通 | 模拟器稳定，VPT 1x 推理，记录 FPS/VRAM |
| M1 | 僵尸竞技场 | 20 次可复现 reset，随机出生和朝向 |
| M2 | 基线与指标 | 50 个 seed 的 zero-shot 评估报告 |
| M3 | 首次 PPO | 1M steps，训练曲线，stage-1 gate |
| M4 | 课程扩展 | 只有 M3 通过后才开始骷髅阶段 |

详见 [docs/roadmap.md](docs/roadmap.md)。

## 致谢

- [MineStudio](https://github.com/CraftJarvis/MineStudio)：模拟器、数据、模型和在线 RL 基础设施
- [MineRL](https://github.com/minerllabs/minerl) 与 [Project Malmo](https://github.com/microsoft/malmo)：底层 Minecraft 环境
- [OpenAI VPT](https://github.com/openai/Video-Pre-Training)：预训练 foundation model

## License

MIT，见 [LICENSE](LICENSE)。
