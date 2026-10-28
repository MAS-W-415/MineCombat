# MineCombat

**从僵尸到末影龙：Minecraft 战斗的课程强化学习。**

MineCombat 是一个基于 [MineStudio](https://github.com/CraftJarvis/MineStudio) 和 [VPT](https://github.com/openai/Video-Pre-Training) 的研究项目。智能体直接从 128×128 RGB 画面学习控制原生键鼠动作——视角、移动、攻击、跳跃、冲刺。长期目标是端到端击败末影龙，第一个里程碑是可复现的僵尸战斗场景。

[English README](README.md)

## 当前状态

`M0 — 环境打通` / `M1 — 僵尸竞技场`

仓库目前包含环境骨架、奖励实现、评估脚本和单卡训练配置。基线与第一次 PPO 实验会在 [Issues](https://github.com/MAS-W-415/MineCombat/issues) 中跟踪。

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
| 0 | 环境 | 稳定 reset、渲染、VPT 推理、日志 | 进行中 |
| 1 | 僵尸 | 接近、瞄准、攻击、躲避伤害 | 下一步 |
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
r_t = 0.1 * Δdamage_dealt
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

本地 RTX 4060 Laptop 只有 8 GB VRAM，所以第一版 PPO 配置刻意保守：

- 1 个 rollout worker，8–12 个环境
- 每卡 batch size 1
- context length 64
- VPT 1x

MineStudio 官方配置默认需要多张 GPU。单卡配置见 `configs/ppo/zombie_vpt1x.yaml`；正式长跑建议放到 4090 或云端。

```bash
bash scripts/train_zombie.sh
```

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
