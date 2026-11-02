# HH1 Tactical Conditioning — GPU Training Runbook

**Project:** MineCombat
**Branch:** `feat/tactical-rewards`
**Starting Commit:** `39af1eb`
**Status:** Engineering + 10k validation complete (see `docs/hh1-validation.md`); longer behavioral run pending
**Owner:** GPU Training & Evaluation
**Scope:** HH1 Conditioned VPT with Tactic-Specific Rewards

---

## 1. Objectives

本阶段目标是验证 MineCombat 的条件化底层策略训练机制，并探索不同战术条件能否产生可重复的执行行为差异。

具体分为三个层次：

1. **Engineering Validation**：确保 Conditioned VPT 能在真实 MineStudio 环境中完成采样、训练、Checkpoint 保存和恢复。
2. **Conditioning Validation**：确认不同 `tactic_id` 对模型动作分布产生可测量影响。
3. **Behavioral Validation**：确认不同 `tactic_id` 在真实 Minecraft 战斗中产生合理、可重复的行为差异。

三者必须明确区分。

**训练成功不等于条件化成功；条件化参数发生变化，也不等于模型学会了实际战术。**

本阶段仅研究底层战术条件学习。HH2 High-Level PPO 的真实环境训练不属于本手册的直接执行范围。

## 2. Current Architecture

### 2.1 Data Flow

```text
Minecraft / MineStudio
        |
        v
CombatRewardCallback
        |
        |  Base Combat Reward
        v
EpisodeTacticObservationWrapper
        |
        |  Observation + tactic_ids
        v
TacticalRewardWrapper
        |
        |  Base Reward + Tactical Auxiliary Reward
        v
Conditioned VPT
        |
        |  Native Minecraft Actions
        v
MineStudio PPO Trainer
        |
        v
Checkpoint + Training Metrics
```

### 2.2 Tactical Conditions

| ID  | Candidate Tactic | Auxiliary Reward                   |
| --- | ---------------- | ---------------------------------- |
| 0   | Aggressive       | Additional reward for damage dealt |
| 1   | Defensive        | Additional penalty for HP lost     |

第一版使用两个候选战术，而不是原来的四个。

这些编号仅表示训练条件。当前代码没有显式动作脚本，不保证最终学到对应语义。

### 2.3 Reward Definition

基础任务奖励由 `configs/env/zombie_stage1.yaml` 定义：

```text
r_base =
    0.01 * delta_damage_dealt
  - 0.20 * delta_hp_lost
  - 0.001
  + 10.0 * kill
  - 5.0 * death
```

战术辅助奖励：

```text
Tactic 0:
    r_aux = +0.005 * delta_damage_dealt

Tactic 1:
    r_aux = -0.1 * delta_hp_lost
```

最终低层 PPO 训练奖励：

```text
r_training = r_base + r_aux
```

`damage_dealt` 使用 Minecraft 的统计计数单位，即约 10 个统计点对应 1 HP 的伤害。

基础奖励中的裁剪发生在 `CombatRewardCallback`。当前战术奖励在此基础上相加，不再自动套用相同裁剪区间，需在日志中检查最终奖励分布。

**重要：不要把辅助奖励带来的训练 Reward 增长直接解释为战斗能力提升。** 最终行为评估必须使用共同的基础战斗指标。

---

## 3. Common Preflight and HH1 Config

Run the shared preflight in [training-common.md](training-common.md) first. It
covers hardware, JDK 8, the Hugging Face mirror, engine patches, Ray, tests,
simulator smoke, damage validation, checkpoints, and cleanup.

Then check the HH1 config:

```bash
PYTHONPATH=src python -m minecombat.train.ppo \
  --conditioned \
  --num-tactics 2 \
  --sample-tactic-per-episode \
  --tactical-rewards \
  --dry-run
```

Confirm the output contains:

```yaml
harness:
  conditioned: true
  num_tactics: 2
  tactic_id: 0
  sample_tactic_per_episode: true
  tactical_rewards: true
```

`--dry-run` only checks argument parsing and config generation; it does not
start Minecraft or prove that PPO training works.

## 4. Environment Preflight

The simulator smoke test and damage-signal validation are in
[training-common.md](training-common.md#preflight-checks). Run them before the
HH1 GPU smoke test.

## 5. HH1 GPU Smoke Test

### 5.1 Purpose

使用最小训练配置验证：

- Ray Worker 可以启动。
- Conditioned VPT 可以加载。
- `tactic_ids` 能正确进入训练模型。
- TacticalRewardWrapper 正常计算奖励。
- PPO 能执行反向传播和优化器更新。
- 模型及优化器 Checkpoint 能保存。

不要求胜率提升。

### 5.2 Start Ray

Use the shared Ray commands in
[training-common.md](training-common.md#ray). Check that JDK 8, the custom
resources, and any stale processes are handled there.

### 5.3 Run 512-Step Smoke Test

```bash
bash scripts/train_zombie.sh \
  --conditioned \
  --num-tactics 2 \
  --sample-tactic-per-episode \
  --tactical-rewards \
  --smoke \
  --save-path experiments/h1_tactical_smoke \
  --name h1_tactical_smoke
```

当前 Smoke 配置：

| Parameter                      | Value |
| ------------------------------ | ----: |
| Iterations                     |     1 |
| Batches per iteration          |     2 |
| Fragment length                |   256 |
| Environments                   |     1 |
| Effective target rollout steps |  ~512 |

实际采样数量以日志为准。

### 5.4 Monitor Training

在另一终端运行：

```bash
nvidia-smi
```

检查：

- GPU 显存峰值。
- 是否存在 CUDA OOM。
- 是否发生 Ray Worker 崩溃。
- 是否完成至少一次 PPO 更新。
- Policy Loss、Value Loss 是否为有限值。
- Checkpoint 是否正常生成。

如果训练反复崩溃，不应通过增加训练步数解决。

### 5.5 Locate Checkpoints

```bash
find experiments/h1_tactical_smoke \
  -type f \
  \( -name "model.ckpt" -o -name "optimizer.ckpt" \)
```

记录实际保存路径。

### 5.6 Inspect Conditioned Parameters

将下面命令中的路径替换为实际 Checkpoint：

```bash
CKPT=/actual/path/to/model.ckpt

python - "$CKPT" <<'PY'
import sys
import torch

state = torch.load(
    sys.argv[1],
    map_location="cpu",
    weights_only=True,
)

if isinstance(state, dict) and "model" in state:
    state = state["model"]

if not isinstance(state, dict):
    raise TypeError("Unexpected checkpoint structure")

keys = [k for k in state if "conditioner." in k]

print("Conditioner keys:", keys)

if not keys:
    raise RuntimeError("No conditioned parameters found")

for key in keys:
    value = state[key]
    if torch.is_tensor(value):
        print(
            key,
            tuple(value.shape),
            "finite=", bool(torch.isfinite(value).all()),
            "norm=", float(value.float().norm()),
        )
PY
```

必须确认：

- 包含 Embedding 和 Projection 权重。
- 参数形状正确。
- 参数值没有 NaN / Inf。

仅有非零权重不等于发生了有效学习。理想情况下，还应比较训练前后同名参数，或在训练日志中记录相关梯度和参数变化。

### 5.7 Smoke Acceptance Criteria

- [ ] Ray 与 Minecraft 正常启动。
- [ ] Conditioned VPT 完成采样。
- [ ] PPO 至少完成一次优化更新。
- [ ] 没有未解决的 NaN / Inf。
- [ ] Checkpoint 保存成功。
- [ ] Checkpoint 中存在条件化参数。
- [ ] 相关权重能够加载。
- [ ] 未发现奖励统计异常。

只有全部满足，才进入下一阶段。

---

## 6. 10K-Step Diagnostic Training

### 6.1 Objective

验证短时间内的训练稳定性，并观察战术条件模块是否开始影响策略。

这仍然属于诊断实验，不应作为最终性能结论。

### 6.2 Training Command

```bash
bash scripts/train_zombie.sh \
  --conditioned \
  --num-tactics 2 \
  --sample-tactic-per-episode \
  --tactical-rewards \
  --smoke \
  --num-iterations 20 \
  --save-path experiments/h1_tactical_10k \
  --name h1_tactical_10k
```

解释：

`--smoke` 将每轮训练缩小为 2 个 Fragment，随后 `--num-iterations 20` 将训练轮数设为 20。

理论采样量：

```text
20 iterations × 2 fragments × 256 steps
= 10,240 environment steps
```

实际完成步数应以日志为准。

### 6.3 Metrics to Track

至少记录：

| Metric                    | Purpose           |
| ------------------------- | ----------------- |
| Total environment steps   | 训练进度          |
| PPO updates               | 实际优化次数      |
| Policy Loss               | 策略优化行为      |
| Value Loss                | Critic 拟合情况   |
| Entropy                   | 策略随机性        |
| Base Reward               | 共同战斗任务表现  |
| Tactical Auxiliary Reward | 战术辅助奖励贡献  |
| Training Reward           | 实际 PPO 优化信号 |
| Episode length            | Episode 持续时间  |
| Death / Kill events       | 战斗结果          |
| GPU VRAM / RAM            | 运行资源需求      |

当前 Wrapper 已在 `info` 中记录：

```text
reward_base
reward_tactical_aux
reward_training
tactic_id
```

但这些字段不一定被 MineStudio 默认 Logger 自动汇总。

**如果训练日志没有出现这些指标，需要增加 Logger 或 Episode Statistics 集成。不能直接假设 Wrapper 写入 `info` 就等于 W&B 已记录。**

### 6.4 Tactic Sampling Check

目前 `--sample-tactic-per-episode` 在每个 Episode 开始时随机选取一个战术。

因此必须记录：

- Tactic 0 的 Episode 数量。
- Tactic 1 的 Episode 数量。
- 两种战术各自贡献的环境步数。
- 两种战术各自的基础 Reward 和辅助 Reward。

短训练不保证战术样本完全均衡。

此外，当前 Wrapper 使用独立随机数生成器；多环境训练时需要检查 Worker 间种子是否正确区分，防止多个环境重复产生相同战术序列。

### 6.5 Acceptance Criteria

- [ ] 约 10K 环境步顺利完成。
- [ ] 没有持续性训练崩溃。
- [ ] 参数更新与 Checkpoint 正常。
- [ ] 两种战术均实际获得采样数据。
- [ ] 基础奖励与辅助奖励能够分别统计。
- [ ] 没有明显异常的奖励重复计数。
- [ ] 没有模型数值发散。

如果训练稳定但战术差异不明显，可以继续实验；如果训练链路不稳定，应先修复工程问题。

---

## 7. Tactical Conditioning Validation

这是 HH1 最关键的验证步骤。

### 7.1 Parameter-Level Check

检查条件化模块参数是否参与训练：

```text
conditioner.embedding.weight
conditioner.projection.weight
```

重点比较：

- 训练前后参数差异。
- Projection 参数范数。
- 梯度是否流向条件化模块。
- 是否出现 NaN / Inf。

注意 Projection 采用零初始化。早期 Embedding 的有效梯度可能较弱，因此不能单凭首次更新的 Embedding 变化判断训练失败。

### 7.2 Action Distribution Sensitivity

对同一个模型、同一观察和相同循环记忆，分别输入：

```text
tactic_id = 0
tactic_id = 1
```

比较两种条件下的动作分布。

建议指标：

- Action Logit Difference
- 平均动作概率差异
- KL Divergence
- 可选：Jensen-Shannon Divergence

测试时必须：

1. 固定模型权重。
2. 使用完全相同的观察。
3. 使用相同的初始循环状态。
4. 不额外推进一次 VPT 记忆。
5. 对多个观察样本进行统计。

结果解释：

- 差异几乎为零：条件化可能被忽略。
- 差异非零：条件化已影响输出。
- 差异明显但行为无改善：仍然不能说明战术学习成功。

### 7.3 Implementation Status

当前仓库已经实现战术 Embedding 和 Projection，但尚无专门的条件化动作分布对照脚本。

因此本检查需要补充独立诊断脚本及测试。

**在该诊断工具可用前，不应宣称“战术条件影响已通过验证”。**

---

## 8. Tactical Behavior Evaluation

### 8.1 Objective

评估是否真正形成不同的底层执行偏好，而不是仅产生不同模型参数。

### 8.2 Evaluation Groups

使用同一个训练后的 Conditioned VPT Checkpoint：

| Group | Tactical Condition          |
| ----- | --------------------------- |
| A     | Fixed Tactic 0 — Aggressive |
| B     | Fixed Tactic 1 — Defensive  |

可以额外设置：

| Control | Description                               |
| ------- | ----------------------------------------- |
| C       | Zero-shot VPT                             |
| D       | Flat VPT + PPO                            |
| E       | Conditioned VPT without auxiliary rewards |

其中 C、D、E 用于判断新训练机制相比原方法是否有收益。

### 8.3 Fair Evaluation Protocol

所有实验组必须保持：

- 相同 Minecraft 环境配置。
- 相同出生距离分布。
- 相同随机种子。
- 相同最大 Episode 长度。
- 相同确定性或随机推理设置。
- 相同基础战斗指标计算方法。

初步建议每组至少 50 个固定种子 Episode：

```text
seed_start = 0
episodes = 50
max_steps = 600
```

可在发现有效差异后扩大评估样本，并采用多个训练种子复现。

### 8.4 Evaluation Entry Point

The unified CLI now supports conditioned policies:

```bash
python scripts/eval.py \
  --policy conditioned \
  --checkpoint experiments/<hh1>/checkpoints/<timestamp>/<update>/model.ckpt \
  --num-tactics 2 --tactic-ids 0 1 \
  --episodes 50 --seed-start 0 --max-steps 600 --deterministic
```

The runner holds one tactic fixed per episode, reuses the same low-level
weights, and writes a separate run directory per tactic. The older
`eval_baseline.py` and `eval_conditioned.py` scripts remain as compatibility
wrappers.

### 8.5 Behavioral Metrics

至少记录：

| Metric              | Explanation                          |
| ------------------- | ------------------------------------ |
| Win Rate            | 击杀僵尸的 Episode 比例              |
| Death Rate          | 玩家死亡比例                         |
| Mean HP Lost        | 平均累计 HP 损失                     |
| Damage Dealt        | 平均造成伤害                         |
| Damage Ratio        | 造成伤害与受到伤害之比，注意统计单位 |
| Attack Frequency    | 攻击键动作出现频率                   |
| Mean Episode Length | 平均持续步数                         |
| Time-to-Kill        | 成功击杀所需步数                     |

还应保存代表性视频。

### 8.6 Expected Behavioral Tendencies

候选假设：

**Aggressive：**

- 更积极地执行攻击。
- 倾向于提高有效伤害。
- 可能承担更多 HP 损失。

**Defensive：**

- 倾向于减少受到的伤害。
- 可能降低攻击频率或增加战斗持续时间。
- 仍应具备完成击杀任务的能力。

不能仅因为 Defensive 的死亡率下降就认定战术成功；如果它完全避免战斗、最终大多数 Episode 超时，也不是理想结果。

最终应综合观察战斗视频和定量指标。

---

## 9. Long-Run Training Decision

不建议在 512 步或 10K 诊断完成前直接启动大规模训练。

若前面检查全部通过，可以采用逐级扩大预算的策略。

| Stage      | Suggested Environment Steps | Goal                           |
| ---------- | --------------------------: | ------------------------------ |
| Smoke      |                        ~512 | 验证训练闭环                   |
| Diagnostic |                        ~10K | 检查数据、梯度和稳定性         |
| Pilot      |                     20K–50K | 观察战术条件是否产生方向性差异 |
| Extended   |                   100K–300K | 形成可评估的候选策略           |
| Longer Run |              视实际结果决定 | 进一步验证性能与稳定性         |

这些是建议预算，不是收敛保证。

长训练应在以下条件下开展：

- 训练稳定。
- 条件化模块确实受到更新。
- 两种战术的数据采集比例合理。
- 已建立有效的评估接口。
- 不存在明显的奖励漏洞。
- 基础战斗奖励可以与辅助奖励分离统计。

### 9.1 Training Experiments

建议优先比较：

**Experiment A — Conditioning Only**

```text
Conditioned VPT
No tactical auxiliary reward
Two tactic IDs
```

**Experiment B — Conditioning + Auxiliary Rewards**

```text
Conditioned VPT
Two tactic IDs
Tactic-specific auxiliary rewards
```

两组尽可能使用相同的模型初始化、环境配置及训练预算。

如果 B 比 A 更容易形成合理的战术差异，说明辅助奖励机制可能有效。

如果两组差异不明显，需要重新检查奖励权重、数据采集和策略学习机制，而不是直接假设增加步数一定能解决。

---

## 10. Checkpoint and Experiment Management

### 10.1 Naming Convention

建议使用独立实验目录：

```text
experiments/
    h1_tactical_smoke/
    h1_tactical_10k/
    h1_conditioned_only/
    h1_tactical_pilot/
    h1_tactical_extended/
```

避免不同实验覆盖同一 Checkpoint。

### 10.2 Required Metadata

每次实验至少记录：

```yaml
experiment_name: h1_tactical_pilot
git_commit: "<actual commit SHA>"
model: conditioned_vpt_1x
num_tactics: 2
tactical_rewards: true
sample_tactic_per_episode: true
training_steps: "<actual value>"
seed: "<actual seed>"
gpu: "<actual GPU>"
checkpoint: "<actual path>"
```

同时保存实际解析后的 PPO 和环境配置。

### 10.3 Resume Safety

恢复训练之前必须确认：

- 模型结构与 Checkpoint 匹配。
- `num_tactics` 一致。
- 使用了正确的奖励配置。
- 优化器状态是否需要恢复。
- 训练步数与日志能连续追踪。
- 恢复前后使用了相同的环境版本。

不得将不同训练阶段的 Checkpoint 混用而不记录来源。

### 10.4 Artifact Policy

保留：

- 模型 Checkpoint。
- 优化器 Checkpoint（需要恢复训练时）。
- 实际训练配置。
- W&B 或训练日志。
- 评估 JSON/CSV。
- 代表性战斗视频。

大型 Checkpoint 不直接提交 Git 仓库。

---

## 11. Failure Handling

### 11.1 CUDA Out of Memory

优先：

1. 确认没有其他训练任务占用 GPU。
2. 检查显存峰值。
3. 降低环境并行数。
4. 检查训练 Batch 与梯度累积配置。

不要只依靠频繁清空 CUDA Cache 掩盖持续性内存问题。

### 11.2 Ray Worker Crash

检查：

- Ray 自定义资源。
- Java / Minecraft 进程。
- CPU 和 RAM。
- 环境初始化日志。
- 是否存在异常遗留的 Simulator 进程。

### 11.3 Non-Finite PPO Loss

检查：

- Reward 是否出现异常大值。
- 输入特征是否有限。
- 条件化参数是否出现 NaN / Inf。
- Optimizer 和梯度范数。
- `tactic_ids` 的范围与张量维度。

### 11.4 Tactic Ignored by Policy

如果不同战术编号下输出基本相同：

1. 确认 Rollout 记录了正确的 `tactic_ids`。
2. 确认 PPO 训练时使用的 ID 与采样时一致。
3. 检查 Conditioner 梯度。
4. 检查 Projection 是否发生更新。
5. 检查奖励差异是否足以影响优化目标。
6. 检查不同战术的数据量是否合理。

### 11.5 Reward Hacking

典型现象包括：

- 只攻击一次然后一直逃跑。
- 长时间旋转或停滞。
- Defensive 完全回避战斗。
- 训练 Reward 增加，但击杀率下降。

必须通过视频与基础战斗指标判断，不应只看训练 Reward。

---

## 12. Final Deliverables

队友完成本阶段后，需要提供以下结果。

### 12.1 Environment Report

- GPU 型号、显存、系统 RAM。
- Python、PyTorch、CUDA、JDK、Ray 版本。
- Git Commit。
- 实际运行配置。

### 12.2 Smoke Test Report

- 512 步 Smoke 运行结果。
- PPO 实际更新次数。
- 是否出现错误。
- 峰值 GPU 显存。
- Checkpoint 路径。
- 条件化参数检查结果。

### 12.3 Diagnostic Training Report

- 10K 步训练曲线。
- Base Reward、Auxiliary Reward、Training Reward。
- 两种战术的数据采样占比。
- Policy Loss、Value Loss、Entropy。
- 参数更新和数值稳定性。

### 12.4 Behavioral Evaluation Report

在评估脚本具备完整支持后，提交：

- Aggressive 与 Defensive 的固定种子评估。
- 各项战斗指标对比表。
- 至少数段代表性战斗录像。
- 失败案例分析。
- 是否观察到可重复的战术行为差异。

### 12.5 Final Conclusion

报告必须明确回答：

1. HH1 Conditioned PPO 训练链路是否正常？
2. 战术条件是否影响动作分布？
3. 两种战术是否产生可重复的执行差异？
4. 是否有证据表明战术奖励机制优于仅使用随机战术编号？
5. 是否具备进入 HH2 高层 PPO 训练的条件？

如果答案是否定的，也应保留实验结果并说明失败原因。

---

## 13. Handoff to HH2 High-Level PPO

HH1 训练完成并不意味着可以直接运行 HH2。

进入 HH2 之前，需要确认：

- Conditioned VPT Checkpoint 可加载。
- 战术编号与高层输出空间一致。
- 已验证至少初步可区分的底层行为。
- 高层环境交互循环可使用真实 Minecraft Reward。
- High-Level Rollout 能正确记录持续时间。
- PPO 采样时的旧 Log Probability 和旧 Value 正确保留。
- 高层训练期间底层 VPT 可以冻结。
- Episode 结束和时间截断处理正确。

HH2 的目标是：

```text
Visual Features
      |
      v
High-Level Actor-Critic
      |
      v
Tactical Decision
      |
      v
Frozen Conditioned VPT
      |
      v
Minecraft Action
      |
      v
Environment Reward
      |
      v
High-Level SMDP PPO Update
```

HH2 将使用共同的真实战斗奖励，而不是简单将 HH1 的战术辅助奖励作为高层优化目标。

HH2 的真实在线训练入口、模型同步与专用评估脚本需要另行集成和验证。

---

## 14. Completion Checklist

### Engineering

- [ ] 环境与依赖预检完成。
- [ ] 11 个回归测试通过。
- [ ] 512 步 GPU Smoke 完成。
- [ ] PPO 优化器实际执行更新。
- [ ] Conditioned Checkpoint 保存、加载成功。
- [ ] 10K 步诊断训练稳定。

### Tactical Conditioning

- [ ] 确认两种战术都参与训练。
- [ ] 条件化参数实际更新。
- [ ] 同状态不同战术的动作分布对照完成。
- [ ] 两种战术的实际行为评估完成。
- [ ] 排除明显的奖励漏洞。

### Reproducibility

- [ ] Git Commit 和配置已保存。
- [ ] 模型与日志路径已记录。
- [ ] 固定种子评估完成。
- [ ] 训练视频和异常案例已整理。
- [ ] 实验结论已提交。

**Exit Criterion:** HH1 不仅能够完成训练，而且具备可复现的条件化行为评估结果，才能作为高层战术选择策略的可靠底层基础。

---

**Reference**

- `configs/env/zombie_stage1.yaml`
- `configs/ppo/zombie_vpt1x.yaml`
- `src/minecombat/train/ppo.py`
- `src/minecombat/train/generators.py`
- `src/minecombat/policies/harness/tactic_rewards.py`
- `src/minecombat/policies/harness/tactical_reward_wrapper.py`
- `src/minecombat/policies/harness/conditioned_vpt.py`
- `docs/1m-run.md`
- `docs/m3-ppo-smoke.md`
