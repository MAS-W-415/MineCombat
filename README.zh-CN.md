# MineCombat

**从僵尸到末影龙：Minecraft 战斗的课程强化学习。**

MineCombat 是一个基于 [MineStudio](https://github.com/CraftJarvis/MineStudio) 和 [VPT](https://github.com/openai/Video-Pre-Training) 的研究项目。智能体直接从 128×128 RGB 画面学习控制原生键鼠动作——视角、移动、攻击、跳跃、冲刺。长期目标是端到端击败末影龙，第一个里程碑是可复现的僵尸战斗场景。

[English README](README.md)

## 当前状态

`M0 — 环境打通`、`M1 — 僵尸竞技场`、`M2 — 基线与指标` 已完成。
`M3 — 首次 PPO` 进行中：单卡训练链路、10k smoke、dense damage 信号和
50-seed checkpoint 评估都已跑通。1M 长跑需要 24 GB GPU / 64 GB RAM 的机器；
本地 RTX 4060 单环境约 45 steps/s。

主要结果：

- [M2 基线](docs/m2-baseline.md)：随机策略 0% 胜率 / 96% 死亡率；zero-shot VPT 2% 胜率 / 28% 死亡率。
- [M3 PPO smoke](docs/m3-ppo-smoke.md)：10k steps、20 次 update、峰值显存 6.3 GiB。
- [Dense damage](docs/m3-dense-damage.md)：10k 时首次出现非零伤害比（0.0897）。
- [1M 运行手册](docs/1m-run.md)：下一台机器上的完整命令与前置检查。

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
python scripts/eval_baseline.py --policy random --episodes 5
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

- **H1**：从 VPT 1x 做 PPO，能否提升僵尸胜率并降低掉血？
- **H2**：僵尸预训练策略迁移到骷髅，是否比直接从 VPT 微调更省样本？
- **H3**：在 flat policy 足够强之后，高层 Harness 是否能提升末影龙长程任务的胜率？

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
