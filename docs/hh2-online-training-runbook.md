# MineCombat HH2 — High-Level PPO GPU Training & Acceptance Runbook

**Project:** MineCombat
**Branch:** `feat/h2-online-training`
**Status:** Local Mock Tests Completed / GPU Validation Pending
**Training Target:** HH2 High-Level Tactical Actor-Critic
**Low-Level Model:** Frozen HH1 Conditioned VPT
**Environment:** MineStudio Zombie Arena — Stage 1

---

## 1. Objectives

### 1.1 Primary Goal

验证 MineCombat 的 HH2 分层强化学习系统能否在真实 Minecraft 环境中完成以下闭环：

1. 加载训练后的 HH1 Conditioned VPT Checkpoint。
2. 冻结 HH1 底层策略。
3. 由 HH2 High-Level Actor-Critic 选择战术编号。
4. 底层 VPT 根据战术编号执行 Minecraft 动作。
5. 采集真实环境的 Reward 和高层 Semi-MDP Transition。
6. 使用 Semi-MDP PPO 更新高层策略。
7. 保存并重新加载 HH2 Checkpoint。

### 1.2 Validation Levels

将验收分为三个级别：

| Level                         | Goal                       | Required Evidence                |
| ----------------------------- | -------------------------- | -------------------------------- |
| L1 — Engineering Smoke        | 真实训练链路能够运行       | 环境采样、PPO 更新、Checkpoint   |
| L2 — Hierarchical Correctness | 高低层交互与数据记录正确   | 战术传递、持续时间、奖励、边界   |
| L3 — Behavioral Effectiveness | 高层能够学习合理的战术选择 | 固定种子评估、对照实验、战斗表现 |

本次 GPU Smoke Test 优先完成 L1 和 L2。

**L1 通过不代表 HH2 已经学会高层战术选择。**

---

## 2. Current System Architecture

### 2.1 Overall Data Flow

```text
Minecraft Observation
          |
          v
VisualGridFeatureAdapter
          |
          |  192-D Feature Vector
          v
HH2 High-Level Actor-Critic
          |
          |  tactic_id ∈ {0, 1}
          v
FixedIntervalGate
          |
          |  One decision every 20 steps
          v
Frozen HH1 Conditioned VPT
          |
          |  Minecraft Native Action
          v
Minecraft Environment
          |
          |  Base Combat Reward
          v
HighLevelRolloutSession
          |
          v
TacticalRolloutCollector
          |
          |  Semi-MDP Transitions
          v
High-Level PPO Update
          |
          v
HH2 Actor-Critic Checkpoint
```

### 2.2 Main Components

| File                    | Responsibility               |
| ----------------------- | ---------------------------- |
| `high_level_factory.py` | 加载冻结 HH1，组装 HH2 Harness |
| `visual_features.py`    | 将图像提取为高层特征         |
| `tactical_policy.py`    | HH2 Actor-Critic              |
| `runtime.py`            | 高低层策略调度               |
| `high_level_session.py` | 动作、奖励与轨迹记录         |
| `tactical_rollout.py`   | Semi-MDP Transition          |
| `episode_collector.py`  | 真实环境 Episode 采样        |
| `high_level_ppo.py`     | 高层 PPO / SMDP GAE          |
| `high_level_trainer.py` | 训练循环与 Checkpoint        |
| `train/high_level.py`   | 真实训练命令行入口           |

### 2.3 Training Characteristics

当前实现采用：

- 单个 Minecraft Environment。
- 同步 On-Policy PPO 训练。
- 冻结 HH1 Conditioned VPT。
- 只更新 HH2 Actor-Critic。
- 固定间隔战术决策。
- 基于原始战斗任务奖励训练 HH2。
- 每次 PPO 更新后丢弃旧 Rollout。

**HH2 当前不依赖 Ray，也不需要启动 `scripts/start_ray.sh`。**

---

## 3. Prerequisites

### 3.1 Hardware

推荐环境：

- Linux x86-64
- NVIDIA GPU，优先 RTX 4090 24 GB
- 64 GB 或更多系统 RAM
- 足够的磁盘空间保存模型和日志

24 GB 显存是推荐配置，不保证一定不会 OOM。实际峰值显存必须通过 GPU Smoke Test 测量。

### 3.2 Software

需要安装：

- Python 3.10
- 与 GPU 匹配的 CUDA/PyTorch
- JDK 8
- MineStudio
- Gymnasium
- MineCombat 项目依赖

HH2 单环境同步训练不需要独立启动 Ray 服务。

### 3.3 HH1 Checkpoint Requirement

HH2 训练必须基于真实 HH1 Checkpoint。

Checkpoint 应满足：

- Conditioned VPT 使用 2 个战术编号。
- 包含完整模型权重。
- 可以恢复 `ConditionedVPTPolicy`。
- HH1 已完成真实 GPU Smoke Test。
- 不存在严重的战术条件传递错误。

如果 HH1 尚未形成可区分的行为，可以进行 HH2 工程 Smoke Test，但不能据此评价高层战术学习效果。

---

## 4. Source Code Preflight

### 4.1 Checkout Branch

首次拉取分支：

```bash
git fetch origin
git switch --track origin/feat/h2-online-training
```

若分支已经存在：

```bash
git switch feat/h2-online-training
git pull --ff-only
```

记录版本：

```bash
git rev-parse HEAD
git status -sb
```

确认工作区没有未预期的代码修改。

### 4.2 Activate Environment

```bash
source .venv/bin/activate

python --version
java -version
nvidia-smi
```

如果服务器使用 Conda，应激活实际项目环境，而非照搬 `.venv` 路径。

### 4.3 Verify PyTorch

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
PY
```

必须确认 CUDA 可用。

### 4.4 Run Local Regression Tests

```bash
PYTHONPATH=src python -m pytest \
  tests/test_high_level_cli.py \
  tests/test_high_level_factory.py \
  tests/test_high_level_trainer.py \
  tests/test_episode_collector.py \
  tests/test_high_level_rl.py \
  -v
```

当前相关测试预期为 9 项通过。

然后执行：

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

如果完整测试失败，需要区分依赖缺失、环境问题和代码回归。

---

## 5. HH1 Checkpoint Preflight

### 5.1 Set Checkpoint Path

将以下路径替换为实际 HH1 Checkpoint：

```bash
export HH1_CKPT="/absolute/path/to/HH1/model.ckpt"

test -f "$HH1_CKPT" && echo "HH1 checkpoint found"
sha256sum "$HH1_CKPT"
```

记录 SHA256，确保后续实验使用同一份 HH1 模型。

### 5.2 Inspect Checkpoint Structure

```bash
python - "$HH1_CKPT" <<'PY'
import sys
import torch

checkpoint = torch.load(
    sys.argv[1],
    map_location="cpu",
    weights_only=True,
)

print("Checkpoint type:", type(checkpoint).__name__)

if isinstance(checkpoint, dict):
    print("Top-level keys:", list(checkpoint)[:30])

state = (
    checkpoint["model"]
    if isinstance(checkpoint, dict) and "model" in checkpoint
    else checkpoint
)

if not isinstance(state, dict):
    raise TypeError("Unexpected model state structure")

conditioner_keys = [
    key for key in state
    if "conditioner." in key
]

print("Conditioner keys:")
for key in conditioner_keys:
    print(" ", key)

if not conditioner_keys:
    raise RuntimeError("Conditioner weights not found")
PY
```

需要确认存在对应的 Embedding 和 Projection 权重。

注意：该检查只能证明 Checkpoint 包含相关参数，不能证明已经学会进攻或防御行为。

如果 `weights_only=True` 无法读取文件，应先核查保存端格式和来源，不要直接关闭安全加载限制。

### 5.3 Verify Full HH1 Loading

这是重要的集成风险点。

现有工厂函数通过：

```python
low_policy.load_state_dict(state, strict=True)
```

恢复 HH1 权重。

因此必须检查：

- State Dict 是否存在意外前缀。
- 模型结构与训练时一致。
- `num_tactics=2`。
- Conditioned VPT 能成功恢复。
- 真实 MineStudio 版本与训练时兼容。

**禁止通过随意改成 `strict=False` 来掩盖不匹配的参数。**

---

## 6. HH2 Model Construction Check

正式启动 Minecraft 之前，先独立构建 HH2 Harness。

```bash
PYTHONPATH=src python - "$HH1_CKPT" <<'PY'
import sys

from minecombat.policies.harness.high_level_factory import (
    build_high_level_training_harness,
)

harness = build_high_level_training_harness(
    conditioned_checkpoint=sys.argv[1],
    device="cuda",
    num_tactics=2,
    interval_steps=20,
    grid_size=8,
    hidden_dim=128,
)

low = harness.low_policy
high = harness.tactical_policy

assert not low.training
assert all(not p.requires_grad for p in low.parameters())
assert all(p.requires_grad for p in high.parameters())

assert high.num_tactics == 2
assert high.feature_dim == 192

assert next(low.parameters()).device.type == "cuda"
assert next(high.parameters()).device.type == "cuda"

print("HH1 frozen:", True)
print("HH2 trainable:", True)
print("High-level feature dim:", high.feature_dim)
print("Harness construction passed")
PY
```

### Acceptance

- [ ] HH1 Checkpoint 完整恢复。
- [ ] 底层 VPT 参数全部冻结。
- [ ] 底层 VPT 处于 Eval Mode。
- [ ] 高层 Actor-Critic 参数可训练。
- [ ] 高层、底层位于正确设备。
- [ ] 高层特征维度为 192。
- [ ] 无 CUDA OOM。

这里的冻结检查只能验证构建后的状态。完整训练中仍需确认优化器没有包含底层参数。

---

## 7. HH2 CLI Dry Run

执行：

```bash
PYTHONPATH=src python -m minecombat.train.high_level \
  --hh1-checkpoint "$HH1_CKPT" \
  --device cuda \
  --num-tactics 2 \
  --interval-steps 20 \
  --num-updates 1 \
  --episodes-per-update 1 \
  --max-steps 100 \
  --save-dir experiments/h2_smoke \
  --dry-run
```

预期输出包含：

```json
{
  "num_tactics": 2,
  "interval_steps": 20,
  "num_updates": 1,
  "episodes_per_update": 1,
  "max_steps": 100,
  "device": "cuda",
  "dry_run": true
}
```

以上只是关键字段示例，实际输出还包含其他路径和训练参数。

Dry Run 不会加载模型或检查真实 Checkpoint 的内容，也不会启动 Minecraft。

---

## 8. HH2 GPU Smoke Test

### 8.1 Goal

完成一次真实的高层 PPO 更新，验证端到端训练。

### 8.2 Recommended Smoke Configuration

| Parameter           |  Value |
| ------------------- | -----: |
| num_tactics         |      2 |
| interval_steps      |     20 |
| grid_size           |      8 |
| hidden_dim          |    128 |
| num_updates         |      1 |
| episodes_per_update |      1 |
| max_steps           |    100 |
| learning_rate       | 0.0003 |
| seed                |     42 |

### 8.3 Start Training

```bash
export RUN_DIR="experiments/h2_smoke_001"

mkdir -p "$RUN_DIR"

set -o pipefail

PYTHONPATH=src python -m minecombat.train.high_level \
  --hh1-checkpoint "$HH1_CKPT" \
  --device cuda \
  --num-tactics 2 \
  --interval-steps 20 \
  --num-updates 1 \
  --episodes-per-update 1 \
  --max-steps 100 \
  --learning-rate 0.0003 \
  --seed 42 \
  --save-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/train.log"
```

这是单环境同步训练，不使用 MineStudio 的 Ray PPO Trainer。

### 8.4 Expected Training Flow

```text
Load pretrained VPT
    ↓
Restore trained HH1 conditioner
    ↓
Freeze HH1 parameters
    ↓
Initialize HH2 Actor-Critic
    ↓
Create MinecraftSim
    ↓
Collect one environment episode
    ↓
Generate high-level transitions
    ↓
Run high-level PPO update
    ↓
Save HH2 checkpoint
    ↓
Close Minecraft environment
```

在没有提前终止的情况下：

- 最多执行 100 个 Minecraft 原始环境步。
- 战术决策间隔为 20 步。
- 可能记录约 5 个高层 Transition。
- 完成 1 次高层 PPO Update。

如果玩家提前死亡或完成击杀，实际环境步数和高层 Transition 数量会减少。

### 8.5 Expected Output

正常情况下会看到类似：

```text
[HH2] update=1 steps=100 transitions=5 loss=...
HH2 training complete: ...
```

实际数值取决于环境是否提前终止。

不要强制要求一定是 100 步或 5 个 Transition。

---

## 9. GPU and Process Monitoring

在另一个终端运行：

```bash
nvidia-smi
```

如需保存监控数据：

```bash
nvidia-smi \
  --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
  --format=csv \
  -l 5 > experiments/h2_gpu_monitor.csv
```

监控结束时手动停止该命令。

重点记录：

- GPU 显存峰值。
- GPU 利用率。
- CPU 和系统内存占用。
- Minecraft Java 进程状态。
- 是否发生 CUDA OOM。
- 是否有异常进程残留。

如果发生显存不足，需要先确认底层 VPT 已冻结，采样时处于 `torch.no_grad()` 环境，再考虑模型或运行参数调整。

---

## 10. Checkpoint Validation

### 10.1 Expected Files

Smoke Test 正常完成后，应包含：

```text
experiments/h2_smoke_001/
    high_level_latest.pt
    history.json
    run_config.json
    train.log
```

当前 HH2 Trainer 每次更新都会覆盖 `high_level_latest.pt`。

**当前尚未实现恢复训练功能。不要将已经包含该文件的目录直接用于重新启动。**

### 10.2 Inspect Training History

```bash
python - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])

history = json.loads(
    (run_dir / "history.json").read_text()
)

if not history:
    raise RuntimeError("Empty training history")

for record in history:
    print(json.dumps(record, indent=2))

assert history[-1]["update"] >= 1
assert history[-1]["total_env_steps"] >= 1
assert history[-1]["num_transitions"] >= 1

print("Training history validated")
PY
```

### 10.3 Inspect and Reload HH2 Checkpoint

```bash
PYTHONPATH=src python - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

import torch

from minecombat.policies.harness.tactical_policy import (
    HighLevelTacticalPolicy,
)

run_dir = Path(sys.argv[1])

checkpoint = torch.load(
    run_dir / "high_level_latest.pt",
    map_location="cpu",
    weights_only=True,
)

config = json.loads(
    (run_dir / "run_config.json").read_text()
)

required = {
    "policy",
    "optimizer",
    "update",
    "total_env_steps",
    "total_episodes",
    "train_config",
    "ppo_config",
}

assert required.issubset(checkpoint)

policy = HighLevelTacticalPolicy(
    feature_dim=3 * config["grid_size"] ** 2,
    num_tactics=config["num_tactics"],
    hidden_dim=config["hidden_dim"],
)

policy.load_state_dict(
    checkpoint["policy"],
    strict=True,
)

for name, value in checkpoint["policy"].items():
    if torch.is_tensor(value):
        assert torch.isfinite(value).all(), name

optimizer_state = checkpoint["optimizer"]["state"]

steps = []
for state in optimizer_state.values():
    value = state.get("step")
    if value is not None:
        steps.append(
            int(value.item())
            if hasattr(value, "item")
            else int(value)
        )

assert steps and max(steps) > 0

print("Checkpoint reloaded successfully")
print("PPO update:", checkpoint["update"])
print("Environment steps:", checkpoint["total_env_steps"])
print("Optimizer max step:", max(steps))
PY
```

### 10.4 Acceptance

- [ ] Checkpoint 文件存在。
- [ ] 可以安全加载。
- [ ] 高层 Actor-Critic 参数完整。
- [ ] 参数没有 NaN / Inf。
- [ ] 优化器状态存在且显示已更新。
- [ ] Update 计数正确。
- [ ] 环境步数与训练日志一致。

注意：HH2 Checkpoint 只保存高层模型，不包含完整的 HH1 Conditioned VPT。因此后续推理必须同时提供对应的 HH1 Checkpoint。

---

## 11. Hierarchical Correctness Validation

仅仅看到 PPO Loss 并不能证明分层系统正确。

还需要验证以下逻辑。

### 11.1 Tactical Decision Timing

对于 `interval_steps=20`，应在：

```text
Environment step 1
Environment step 21
Environment step 41
Environment step 61
Environment step 81
...
```

附近出现新的高层决策。

这是在 Episode 未提前结束、Gate 未被额外机制干预时的预期。

### 11.2 Tactical ID Propagation

高层应输出：

```text
TacticalDecision(tactic_id=0)
```

或：

```text
TacticalDecision(tactic_id=1)
```

底层 Conditioned VPT 必须收到同一个战术编号。

当前训练日志没有完整记录每次战术决策，因此需要通过调试日志、断言或额外 Trace 确认：

- 高层实际选择的 ID。
- Gate 是否正确维持战术。
- 底层实际接收的 ID。
- 当前观察是否意外携带另一个 `tactic_ids` 条件。
- MineStudio `get_action()` 是否正确将 `tactical_condition` 传递到模型。

如果无法证明战术条件到达模型前向过程，L2 验收不能判定通过。

### 11.3 Transition Duration

检查：

```text
duration = len(rewards)
```

对于固定 20 步决策间隔：

- 普通完整战术片段通常持续 20 步。
- Episode 结束前的最后片段可能不足 20 步。
- 所有 Transition 的 Duration 总和应等于本次采样的环境步数。

### 11.4 Termination and Truncation

真实终止：

```text
terminated=True
truncated=False
```

不能再进行 Value Bootstrap。

时间限制截断：

```text
terminated=False
truncated=True
```

可以使用终止观察的 Value 作为 Bootstrap，但 GAE 不应跨入下一个 Episode。

当前实现已在单元测试中覆盖相关逻辑，真实环境仍需验证终止标志是否正确传递。

### 11.5 On-Policy Consistency

每次 PPO Update 必须满足：

- 数据来自本次更新前的同一高层策略。
- Rollout 记录旧 Log Probability 和旧 Value。
- 采样期间不更新高层网络。
- PPO 更新后清空旧 Rollout。
- 下一轮使用更新后的策略重新采样。

这些是当前 HH2 Semi-MDP PPO 正确运行的基础条件。

---

## 12. Multi-Update Diagnostic Training

单次更新通过后，再执行一个稍大的诊断实验。

### 12.1 Suggested Configuration

```bash
export RUN_DIR="experiments/h2_diagnostic_001"

mkdir -p "$RUN_DIR"

set -o pipefail

PYTHONPATH=src python -m minecombat.train.high_level \
  --hh1-checkpoint "$HH1_CKPT" \
  --device cuda \
  --num-tactics 2 \
  --interval-steps 20 \
  --num-updates 2 \
  --episodes-per-update 2 \
  --max-steps 200 \
  --learning-rate 0.0003 \
  --seed 42 \
  --save-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/train.log"
```

该配置：

- 2 次 PPO 更新。
- 每次更新采集 2 个 Episode。
- 每个 Episode 最多 200 步。
- 最大总采样量约 800 个环境步。

实际总步数可能小于 800。

### 12.2 Metrics

当前 `history.json` 应提供：

| Metric              | Purpose             |
| ------------------- | ------------------- |
| update              | PPO 更新轮数        |
| total_episodes      | 累计 Episode 数     |
| total_env_steps     | 累计环境步数        |
| update_env_steps    | 本轮采样步数        |
| mean_episode_reward | 平均 Episode Reward |
| total_loss          | 总优化损失          |
| policy_loss         | 策略损失            |
| value_loss          | Critic 损失         |
| entropy             | 策略熵              |
| grad_norm           | 梯度范数            |
| num_transitions     | 高层决策片段数      |

重点观察是否出现非有限数值、梯度异常和采样中断。

**当前日志还缺少按战术分类的选择频率、死亡率、击杀率和每段奖励统计，这些需要后续增加。**

### 12.3 Diagnostic Acceptance

- [ ] 两次 PPO 更新均完成。
- [ ] 四个 Episode 均完成。
- [ ] Transition 数量为正。
- [ ] PPO Loss 为有限值。
- [ ] 环境奖励正常。
- [ ] 高层参数能够更新。
- [ ] 底层 VPT 保持冻结。
- [ ] 第二轮使用新的高层策略采样。
- [ ] 训练完成后环境正确关闭。

---

## 13. Behavioral Evaluation — Future Stage

HH2 训练运行成功后，需要回答：

**高层策略是否学会了根据状态选择不同战术？**

建议增加以下指标：

| Metric                  | Meaning             |
| ----------------------- | ------------------- |
| Tactical Selection Rate | Tactic 0/1 选择比例 |
| Mean Tactical Duration  | 平均战术持续时间    |
| Tactical Switch Rate    | 战术切换频率        |
| Win Rate                | 击杀成功率          |
| Death Rate              | 玩家死亡率          |
| Mean HP Lost            | 平均受到伤害        |
| Mean Damage Dealt       | 平均造成伤害        |
| Mean Time-to-Kill       | 平均击杀时间        |
| Mean Base Reward        | 原始战斗 Reward     |

### 13.1 Recommended Comparison Groups

- HH1 固定 Aggressive。
- HH1 固定 Defensive。
- HH1 随机战术选择。
- HH2 学习得到的战术选择策略。

尽可能让各组使用相同的 HH1 Checkpoint、环境配置和评估种子。

### 13.2 Current Limitation

`eval_conditioned.py` 是 HH1 固定战术评估入口，**不是完整的 HH2 高层策略评估入口**。

目前尚未提供一个能够同时加载：

- HH1 Conditioned VPT Checkpoint；
- HH2 High-Level PPO Checkpoint；

并执行 HH2 确定性评估的专用脚本。

因此 L3 行为验收需要进一步开发评估工具，在工具完善前不得宣称 HH2 已经学会了有效的战术选择策略。

---

## 14. Known Limitations

### 14.1 No Resume Training

当前 `high_level_latest.pt` 可以保存和加载，但 CLI 没有提供 `--resume`。

运行入口也会阻止覆盖已存在的最新 Checkpoint。

训练中断后，不能直接用原目录恢复优化器、随机状态与采样进度。

需要续训功能时，应单独实现并测试。

### 14.2 Single Environment

当前 HH2 是单环境同步实现，适合验证完整流程。

不具备 Ray 多 Worker 的高吞吐量采样能力。

### 14.3 Simplified Visual Features

高层使用：

```text
VisualGridFeatureAdapter(grid_size=8)
```

将 RGB 图像池化为 192 维特征。

这是初版工程实现，不是经过验证的最优视觉表示。

### 14.4 Fixed Decision Interval

当前每 20 步进行一次高层战术决策。

尚未实现 HH3 风险预测或动态决策门控。

### 14.5 Limited Reproducibility

CLI 的 `--seed` 设置 Python、NumPy 和 PyTorch 随机种子。

但 Minecraft 环境还使用 YAML 中的配置种子。当前代码没有完整实现每个 Episode 独立的固定种子控制。

因此不能仅凭设置 `--seed 42` 就认定不同训练运行完全可复现。

### 14.6 No Per-Tactic Logger

HH2 Trainer 目前主要记录 PPO 更新指标。

按战术统计的 Reward、状态、成功率以及选择频率尚未完整接入日志系统。

### 14.7 HH1 Checkpoint Compatibility

由于还没有真实 GPU Checkpoint 的端到端恢复结果，必须检查 MineStudio 保存格式与 `build_conditioned_harness()` 的加载实现是否匹配。

---

## 15. Troubleshooting

### 15.1 HH1 Checkpoint Cannot Load

检查：

- 文件是否存在且完整。
- 是否为 Conditioned VPT 而非 Flat VPT。
- 是否使用 2 个战术。
- State Dict 是否有额外前缀。
- 训练和推理模型结构是否一致。
- MineStudio / PyTorch 版本是否兼容。

不能直接忽略丢失参数。

### 15.2 CUDA Out of Memory

检查：

- 其他 GPU 进程是否占用显存。
- HH1 是否已冻结。
- Rollout 是否使用 `torch.no_grad()`。
- 是否存在未释放的模型副本。
- 实际 VPT 模型显存占用。

当前 HH2 训练器单环境运行，调整 Episode 数量不一定能解决模型本身的显存需求。

### 15.3 Minecraft Environment Crash

检查：

- Java 版本。
- Simulator 初始化日志。
- 环境 YAML。
- 系统 RAM。
- Java 进程残留。
- MineStudio 版本。

若发生异常退出，先清理确认属于该任务的遗留进程，再重新启动。

### 15.4 Tactical Condition Not Received

检查：

- `HarnessRuntime.get_action()`。
- `low_policy.get_action()` 参数传递。
- `ConditionedVPTPolicy.forward()`。
- 是否出现重复的 `tactic_ids` 输入。
- 战术编号是否越界。

这是 HH2 GPU Smoke Test 的重点风险之一。

### 15.5 PPO Loss Is NaN or Inf

检查：

- Reward 是否有限。
- 视觉特征是否有限。
- 旧 Log Probability 是否存在。
- Actor-Critic 参数是否包含 NaN / Inf。
- GAE 结果是否有限。
- 梯度是否异常。

### 15.6 High-Level Policy Always Chooses One Tactic

需要区分：

- 训练样本不足。
- HH1 的战术效果不明显。
- 两种战术的价值差异较小。
- Actor 探索不足。
- 策略真正学会了偏向某个战术。

不能只凭某个战术被频繁选择就判断发生了模式坍缩，也不能仅凭 Loss 下降就判断策略有效。

---

## 16. Deliverables

队友完成 GPU 验收后，应交付以下内容。

### 16.1 Environment Information

```text
Git Commit:
HH1 Checkpoint:
HH1 Checkpoint SHA256:
GPU Model:
GPU VRAM:
System RAM:
Python Version:
PyTorch Version:
CUDA Version:
JDK Version:
MineStudio Version:
```

### 16.2 Smoke Test Results

```text
Environment Started: Yes / No
HH1 Loaded: Yes / No
HH1 Frozen: Yes / No
Tactical Condition Forwarded: Yes / No / Not Checked
PPO Updates Completed:
Episodes Completed:
Environment Steps:
High-Level Transitions:
Checkpoint Saved: Yes / No
Checkpoint Reloaded: Yes / No
Peak GPU Memory:
Critical Errors:
```

### 16.3 Required Artifacts

提交或提供路径：

- `run_config.json`
- `history.json`
- `train.log`
- `high_level_latest.pt`
- GPU 显存监控记录
- HH1 Checkpoint 的来源与校验值
- 实际完成的测试输出
- 错误日志与修复说明

大型 Checkpoint 不直接提交 Git。

### 16.4 Suggested Report Table

| Check                          | Result                    | Notes |
| ------------------------------ | ------------------------- | ----- |
| HH1 Checkpoint Load             | PASS / FAIL               |       |
| Frozen HH1 Verification         | PASS / FAIL               |       |
| HH2 Actor-Critic Initialization | PASS / FAIL               |       |
| Minecraft Environment          | PASS / FAIL               |       |
| Tactical ID Forwarding         | PASS / FAIL / NOT CHECKED |       |
| Episode Collection             | PASS / FAIL               |       |
| Semi-MDP Transition            | PASS / FAIL               |       |
| PPO Optimizer Update           | PASS / FAIL               |       |
| Checkpoint Save                | PASS / FAIL               |       |
| Checkpoint Reload              | PASS / FAIL               |       |
| Multi-Update Stability         | PASS / FAIL               |       |

---

## 17. Final Acceptance Criteria

### L1 — Engineering Smoke

全部满足：

- [ ] 真实 HH1 Checkpoint 成功加载。
- [ ] HH1 参数被冻结。
- [ ] 真实 Minecraft 环境正常运行。
- [ ] 至少完成一个真实 Episode。
- [ ] 至少完成一次高层 PPO 更新。
- [ ] 优化器状态证明参数执行过更新。
- [ ] Loss 和模型参数均为有限值。
- [ ] HH2 Checkpoint 成功保存并重新加载。

### L2 — Hierarchical Correctness

全部满足：

- [ ] 高层实际生成战术 ID。
- [ ] 战术 ID 正确进入底层 VPT。
- [ ] 固定间隔决策符合预期。
- [ ] Transition Duration 与环境步数一致。
- [ ] 真实 Termination / Truncation 正确。
- [ ] PPO 采样和更新满足 On-Policy 约束。
- [ ] HH1 权重未参与 PPO 更新。
- [ ] 多轮更新运行稳定。

### L3 — Behavioral Effectiveness

后续阶段验证：

- [ ] HH2 评估入口已实现。
- [ ] 固定种子评估可复现。
- [ ] HH2 与固定战术策略完成对比。
- [ ] 高层战术选择分布已记录。
- [ ] 有证据表明高层策略能根据状态调整战术。
- [ ] 战斗表现达到预先设定的对照实验目标。

---

## 18. Handoff Decision

最终报告必须给出一个明确状态。

**PASS — GPU Engineering Validated**

L1、L2 均通过。允许进入更长训练及专用评估开发。

**PARTIAL — Pipeline Runs but Validation Incomplete**

训练可以运行，但战术传递、Checkpoint 恢复或数据记录仍有未验证项。只能继续修复或补充验证。

**FAIL — Engineering Blocked**

出现 Checkpoint 不兼容、推理接口失效、持续崩溃、数值发散等严重错误。必须修复后重试。

**BEHAVIOR PENDING**

L1、L2 已通过，但 L3 尚未完成。这是工程验收成功、算法有效性仍待验证的正常状态。

---

## 19. Next Development Tasks

GPU 验收后，按优先级推进：

1. 处理真实 Checkpoint 加载和推理兼容问题。
2. 增加高层战术选择及 Reward 统计。
3. 实现 HH2 Checkpoint 恢复续训。
4. 实现 HH2 固定种子评估入口。
5. 进行固定战术、随机战术和学习战术的对照实验。
6. 根据训练效果决定是否扩展高层状态表示、动态决策周期和 HH3 风险预测。

**Final Principle:** 完整训练链路、正确的分层数据流和真实行为提升是三个独立验收目标，必须分别提供证据。

---

## 20. Reference Files

- `src/minecombat/train/high_level.py`
- `src/minecombat/policies/harness/high_level_factory.py`
- `src/minecombat/policies/harness/high_level_trainer.py`
- `src/minecombat/policies/harness/episode_collector.py`
- `src/minecombat/policies/harness/high_level_session.py`
- `src/minecombat/policies/harness/high_level_ppo.py`
- `src/minecombat/policies/harness/tactical_rollout.py`
- `src/minecombat/policies/harness/tactical_policy.py`
- `src/minecombat/policies/harness/conditioned_vpt.py`
- `configs/env/zombie_stage1.yaml`
- `docs/hh1-tactical-training-runbook.md`

**Document Status:** Ready for HH2 GPU Engineering Handoff. Real GPU validation results are pending.
