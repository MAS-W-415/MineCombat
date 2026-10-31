# MineCombat — Risk-Aware Hierarchical Policy & Harness Architecture

> **Status:** Design proposal v1.0 · 2026-10-31 · **Not yet implemented or empirically validated**  
> **Scope:** Minecraft first-person combat agent, from Zombie Stage 1 toward multi-enemy and long-horizon tasks.  
> **Owners:** Collaborative design; module ownership to be confirmed by both contributors.  
> **Rule:** This document is the architecture baseline for future Harness/Hierarchical Policy changes. Propose changes here before altering public interfaces or the training protocol.

## 中文导读：为什么采用这套架构？

这是一份**面向实现和实验的架构提案**，而不是已经验证的先进算法。我们保留现有 **Flat VPT + PPO** 作为公平对照，另开一条独立的 **分层 Policy / Harness** 研发线：

- **底层 VPT-RL**：唯一负责生成 Minecraft 原生相机和按键动作；接受高层战术条件 `z_t`，通过 PPO 学习如何执行。
- **高层 Tactical Policy**：基于视觉时序信息和风险估计，选择战术条件，而不是发出“按 W / 点击攻击”等逐帧指令；后期使用分层 RL 学习。
- **Risk Critic**：预测未来受伤或死亡风险，辅助战术选择；训练标签可以来自后验游戏日志，但推理不得偷看后台真值。
- **Decision Gate**：决定继续执行当前战术还是重做高层决策；初期可用固定间隔做基线，最终目标是可学习的门控/终止机制。
- **Jev Advisor（可选）**：低频给出结构化战术建议；必须允许完全关闭，并具备超时回退，不能直接控制键鼠。
- **Harness Runtime**：组织这些模块、管理跨帧/跨回合状态、记录训练与评估数据；它本身不是一个手写的攻击状态机。

**开发顺序：H0 接口与 Mock 闭环 → H1 真正条件化 VPT → H2 高层 RL → H3 Risk/Gate → H4 Jev。** 不能把 Fake Policy 测试通过视为真实训练成功，也不能仅给 `get_action` 增加一个未被模型消费的参数就称为“条件化”。

**始终遵循的底线：** 底层模型产生全部逐帧动作；高层只选战术；真实环境数据与特权监督数据分离；Flat PPO 作为稳定对照；新的分层模块必须通过可复现实验验证其价值。如需改变这些原则，先讨论并更新本文档，再修改接口与训练代码。

---

## 0. Decision and evidence level

**Decision:** Retain **Flat VPT + PPO** as the experimental control and introduce an **optional, risk-aware hierarchical Harness** in a separate research/engineering track. The high-level policy chooses a *tactical condition*; the low-level, learned VPT policy alone produces native Minecraft mouse/keyboard actions. The high-level policy and decision/termination gate are intended to become learnable. A risk predictor supplies decision support. Jev, if used, remains a low-frequency, optional advisor.

**Assessment:** This is a technically coherent and promising architecture, **not a demonstrated state-of-the-art result or a proven novel algorithm**. Its building blocks have established precedents: temporal abstraction/options, learned option termination, and pretrained native-action policies. The scientific contribution, if any, must come from a **specific mechanism and controlled experimental improvement**, not from simply combining named components.

Existing repository evidence ([M2](m2-baseline.md), [M3 smoke](m3-ppo-smoke.md), [dense-damage evaluation](m3-dense-damage.md)) shows that the simulator, Flat VPT, and PPO pipeline have run. The reported 50-seed zero-shot VPT win rate is **2%**, while the tested 10k-step dense-damage PPO checkpoint has **0%** win rate. These are *early baselines*, not evidence that hierarchical control is better or that PPO will fail after longer training.

## 1. Reference architecture (normative)

~~~text
MinecraftSim (RGB frames, permitted observation history, previous actions)
         |  step at simulator cadence (~20 Hz in current setup)
         v
Observation / Feature Adapter
  - visual-temporal features h_t (shared VPT features if exposed,
    otherwise an independently validated permitted-observation encoder)
  - recurrent state m_t; no privileged world-state inputs at inference
         |
         +--------------------------+
         |                          |
         v                          v
Risk Critic                    Decision Gate
  q_t = risk(h_{<=t}, z_prev)    g_t = gate(h_{<=t}, q_t, z_prev, state)
  predicted future HP loss,       update now or continue previous tactic
  death probability                |  (fixed cadence in H0; learnable later)
         |                          |
         +-------------+------------+
                       |
             if g_t = 1, query:
                       v
             Tactical Policy π_H
             z_t = π_H(h_{<=t}, q_t, m_H)
             high-level *tactical condition*
                       |
             optional Jev advisor --> fusion / arbitration
             (bounded, low-frequency, typed suggestion,
              never raw key/mouse commands)
                       |
                       v
       Conditioned Low-Level VPT π_L(a_t | o_{<=t}, z_t, m_L)
          camera + buttons in native MineStudio agent action space
                       |
                       v
                 MinecraftSim.step(a_t)
                       |
               reward / termination / log
                       +-----> RL rollout + training (separate pipeline)
~~~

**Required interpretation:** The diagram describes the *target* system. In H0, high-level/Risk/Gate may be mock implementations, and `h_t` extraction may be unavailable. A mocked connection is **not** proof that a real VPT checkpoint conditions on `z_t`.

## 2. Responsibilities and boundaries

| Component | Inputs | Outputs | Trainable? | Must NOT do |
|---|---|---|---|---|
| Observation/Feature Adapter | Permitted RGB, previous action, temporal state | `h_t`, feature state | Optional/frozen initially | Read privileged enemy coordinates as a policy input |
| Tactical Policy `π_H` | `h_t`, risk estimate, high-level history | Tactical condition `z_t`, high-level memory | **Yes**, after low-level conditioning works | Emit direct mouse/button actions |
| Risk Critic | Permitted feature history, candidate/current tactic | Future damage/death-risk estimates | **Yes**, supervised from rollouts | Substitute for PPO value function or read future outcomes at inference |
| Decision Gate | Feature history, risk, elapsed tactic duration | Keep/update decision `g_t` | Fixed baseline → **learnable** | Trigger expensive replanning every frame without a budget |
| Low-Level VPT `π_L` | RGB, action/recurrent history, `z_t` | Native MineStudio action, recurrent memory | **Yes**, PPO fine-tuning | Bypass `z_t` while claiming conditional control |
| Harness runtime | Component instances and states | Scheduling, reset, failure recovery, metrics | Predominantly software | Hide scripted attack/aim/movement macros inside the controller |
| Jev adapter (optional) | Bounded, summarized/typed high-level context | Typed preference/score + uncertainty | External service; optional | Directly operate game actions or become a hard dependency |

**Observation policy:** Stage-1 baseline uses 128×128 RGB and action history. Additional HUD-derived signals require an explicitly documented observation-space change and a matched experimental control. Environment rewards/labels can use backend statistics; deployed Actor/Risk/Gate inputs must not silently use `player_pos`, entity positions, hidden health, kill counters, or future labels unless explicitly authorized in a separate privileged-oracle experiment.

**Low-level ownership of action:** All runtime camera/buttons are sampled or selected by `π_L`. Scripted attacks in `scripts/test_melee.py` are *integration-test oracles*, not the proposed production policy.

## 3. Time scales and inference sequence

At each simulator step `t`:

1. Read observation `o_t` and update the permitted temporal features/recurrent state.
2. Produce or reuse a risk estimate `q_t`. In the first version, risk may be absent (`None`) and the Harness must still function.
3. Gate returns whether to request a new high-level tactic. On reset, the first tactic **must** be initialized; thereafter, retain `z_{t-1}` if the gate stays closed.
4. If updating, evaluate `π_H`; optionally request an advisory typed decision from Jev **without blocking per-frame control**. Apply a defined fallback if the advisor times out or is unavailable.
5. Give `z_t` to the conditioned low-level VPT; obtain `(action, m_L')` using the original MineStudio `agent` action format.
6. Execute one environment step; record reward, `terminated`, `truncated`, `combat_outcome`, and all relevant decisions for replay/debugging.
7. On any reset, clear all episode-local states: low-level recurrent memory, tactical memory, held `z`, gate timers, risk history, advisor cache, and outstanding episode-specific requests.

**Cadence:** Low-level control runs at each environment step; high-level updates are less frequent. H0 may use a fixed frame interval to test interfaces, but it is only a baseline. The target learns termination/update decisions subject to minimum dwell, maximum stale-tactic age, latency and cost constraints. Exact intervals are experimental hyperparameters, not hard-coded architectural truths.

## 4. Tactical condition design

The first training-friendly representation is a **small discrete set of tactical intents**, encoded as a one-hot vector or learned embedding and later replaceable with a continuous condition `z ∈ R^d`. Suggested *semantic* candidates: `engage`, `reposition`, `create_distance`, `seek_opening`. These are **labels for learned behavior preferences**, not programs mapping directly to movement or attacks.

The conditioned VPT must ingest `z_t` through a verified trainable pathway (candidate options: embedding projection added to policy latent, FiLM conditioning, or attention-based conditioning; choose after inspecting MineStudio's actual VPT modules). The preferred H1 experiment is the **smallest working conditioning injection**, preserving pretrained weights and baseline behavior when the conditioning path is disabled. Do not claim a particular injection point is already supported by the current checkpoint.

**Mandatory verification:** For identical observation/history but different `z`, the conditioned architecture must demonstrably expose different action logits *after appropriate training*; the gradient must reach the conditioning parameters; a frozen/unconditioned baseline must behave equivalently to the original VPT. Merely forwarding a `tactical_condition=` Python argument is insufficient.

## 5. RL training responsibilities and sequencing

### A — Flat VPT + PPO (existing independent control)

Keep the current pretrained VPT, MineStudio rollout/trainer and fixed-seed evaluation unchanged as the baseline. The repository's 10k-step checkpoints demonstrate smoke-test completion, **not** adequate combat competence. The planned ~1M-step experiment and Stage-1 gate belong to this track; inspect actual performance before deciding whether to advance the curriculum.

### B — Condition the low-level VPT (new)

Extend the low-level policy interface to accept `z_t`, while **still learning every native mouse/keyboard action by PPO**. Initially use controlled or sampled tactic conditions and compare them against a no-condition (`z` disabled) baseline. Avoid assuming that a high-level learner can improve a low-level actor that has not learned to react to `z`.

Potential failure mode: a learned policy ignores `z` because the environment reward does not distinguish tactics. Require condition sensitivity and meaningful behavioral variation tests before progressing.

### C — Learn a tactical high-level policy (new)

Initially freeze or slowly update `π_L`, then learn `π_H(z | h, q, m_H)` on its **actual slower decision timescale**. A high-level choice lasts `k` environment steps. Treat the high-level transition as semi-Markov: the accumulated discounted reward is

`R_H(t,k) = sum_{i=0}^{k-1} gamma^i * r_{t+i}`

and bootstrap after elapsed duration with `gamma^k`. Do not train it as a naive independent one-step action while discarding variable option durations. The high-level reward should remain comparable to the environment objective; auxiliary penalties/rewards require ablations.

### D — Risk Critic + learned Decision Gate (new)

Train risk predictors on *future* rollout targets, e.g. HP lost over a horizon `H` and probability of death before `H`. Their labels can use environment logs collected **after** the decision timestamp; inference features cannot. Handle censoring at episode termination. Calibrate risks on held-out trajectories, and compare with a no-risk high-level controller.

Start Gate with a fixed-cadence baseline, then learn keep/update behavior or option termination. Impose reasonable switching/compute costs and test for oscillation, collapse to never-update, or always-update. Risk prediction is **distinct** from the PPO value head, which estimates discounted total reward.

### E — Optional Jev fusion (later experiment)

[TypeSafe AI Jev](https://www.typesafeai.org/jev) is a typed-decision model, not a native game-action model. If adopted, supply a concise, privacy-aware textual/structured combat summary and focused choices/scores. Treat output and confidence as *advice* to the tactical layer, with explicit latency budget, asynchronous handling, timeout fallback, uncertainty handling, and service-off mode. **No claim of Jev benefit, image perception performance, or API compatibility is made until tested.**

Conduct paired comparisons: Flat VPT/PPO; Conditioned VPT with a simple high-level policy; Hierarchical policy without Risk; with Risk + Gate; and with Jev enabled. Keep checkpoints and evaluation seeds matched where feasible.

## 6. Runtime API contract (design target, NOT the current MineStudio API)

The current evaluation runner calls `policy.get_action(obs, memory, deterministic=..., input_shape="*")`. The **public Harness Policy must preserve this contract** so it can be plugged into the existing evaluator without rewriting the latter. High-level decisions are *internal* to the wrapper.

~~~python
class HierarchicalCombatPolicy:
    def get_action(self, obs, memory, *, deterministic: bool, input_shape: str):
        # memory owns high/low/gate/risk episode state.
        # 1. extract permitted features
        # 2. predict risk; run gate; optionally refresh tactical z
        # 3. call low-level conditioned policy with z
        # 4. return the native action and updated memory
        return action, next_memory
~~~

Internal typed contracts (names/properties are proposals, to be finalized in H0):

~~~python
@dataclass
class TacticalDecision:
    tactic_id: int                 # or a validated continuous vector later
    confidence: float | None = None

@dataclass
class RiskEstimate:
    expected_hp_loss: float
    death_probability: float
    horizon_steps: int

@dataclass
class HarnessState:
    low_memory: object | None
    high_memory: object | None
    current_tactic: TacticalDecision | None
    steps_since_decision: int
    # gate/risk/advisor episode-local fields are part of state in H0
~~~

These examples intentionally do not specify a tensor shape or VPT latent hook before integration inspection. **The run-time implementation must be batched-safe, reset-safe, deterministic when requested, and compatible with the exact MineStudio action structure**. Retain the original flat policy loader and checkpoint format unless a documented migration is needed.

## 7. Proposed code ownership and repository layout

~~~text
src/minecombat/
  policies/
    vpt.py                    # existing flat inference/checkpoint loading
    harness/
      __init__.py
      contracts.py            # public types and validation
      runtime.py              # orchestration, episode state, gate scheduling
      tactical.py             # learned high-level policy
      risk.py                 # future-risk predictor
      gate.py                 # cadence baseline -> learned termination
      conditioning.py         # trainable z -> VPT feature integration
      jev_adapter.py          # optional, feature-flagged, no hard dependency
  train/
    ppo.py                    # existing flat PPO pipeline; preserve
    hierarchical.py           # future hierarchical rollout/training entry
  eval/
    rollout.py                # existing evaluator; prefer no changes

tests/
  test_harness_contracts.py
  test_harness_runtime.py
  test_harness_conditioning.py
  test_harness_gate.py
  test_harness_risk.py
~~~

**Proposed collaboration split:**

- **Harness/Hierarchical Policy owner:** contracts, runtime, tactical condition, trainable conditioning interface, risk, gate, Jev experiments and Harness diagnostics.
- **Training/Environment owner:** Minecraft/Ray runtime stability, flat PPO, reward and reset logic, large GPU runs, checkpoint/artifact production.
- **Joint ownership:** the VPT model integration point, rollout schemas, conditioned training entry, reproducible comparisons and the merge gate.

Avoid both contributors concurrently rewriting `src/minecombat/train/ppo.py` or the existing `vpt.py` without an agreed interface. Introduce a separate `hierarchical.py` or adapter first. Both owners review changes that modify observation/action or checkpoint contracts.

## 8. Milestones and acceptance tests

| Milestone | Deliverable | Acceptance (no inflated claims) |
|---|---|---|
| **H0 — Interface + runtime skeleton** | Policy-compatible wrapper, typed high/low states, fake high/low/gate implementations | Fake rollout produces valid MineStudio-format actions; first-step tactic initializes; no repeated high-level call when gated off; episode reset clears every state; deterministic path tested. **No claim of learned hierarchical control.** |
| **H1 — Actual conditional VPT** | A trainable `z` injection path + checkpoint/loading adaptation | Gradient reaches condition parameters; action distribution is sensitive to `z` after training; flat baseline remains reproducible; no privileged input. |
| **H2 — High-level RL** | High-level trajectory records and SMDP-aware objective | Variable-duration rewards/discounts verified by unit tests; high-level choices affect deployed policy; preliminary paired evaluation recorded. |
| **H3 — Risk + Gate** | Risk predictor and learned update/termination mechanism | Held-out risk calibration; no leakage; switching cost and gate collapse monitored; ablation against fixed cadence. |
| **H4 — Jev advisory** | Opt-in decision provider and fusion strategy | Offline tests with stubs; service-disabled path identical to non-Jev policy; latency/fallback evaluated; controlled comparison. |

For *every* experimental condition, log configuration, git commit, checkpoint identity, reward definition, observation contract, seeds, episode outcomes, intermediate decisions (`z`, gate, risk), wall-clock latency, and optional video. Use at least the existing **50-seed** benchmark for primary stage comparisons; report uncertainty and do not attribute gains to the Harness without matched controls. A demo video alone is not a research result.

## 9. Risks / falsifiable hypotheses

1. **Conditioning collapse:** VPT ignores high-level `z`. Detect via conditional action-distribution tests and downstream behavior metrics.
2. **Nonstationarity:** Jointly learning high/low policies destabilizes both. Freeze/alternate updates first; compare to end-to-end tuning only after stable baselines.
3. **Credit assignment:** High-level actions span many steps. Use duration-aware returns and test reward bookkeeping.
4. **Risk prediction leakage:** Reward logs/privileged backend data leak into Actor inputs. Separate labels from inference features and test feature allowlists.
5. **Gate pathology:** Update every frame or never update. Track duration histograms and decision cost; compare fixed-cadence Gate.
6. **Latency:** Extra visual inference or external advisory drops effective FPS. Profile the full loop; fail open to the local low-level policy.
7. **No measurable gain:** Hierarchy/Jev may add complexity without improving win rate, HP loss, or efficiency. Publish negative results and retain Flat VPT + PPO as the fair control.

**Hypotheses to test (not assumed true):**

- H-HRL: A *trained, condition-responsive* hierarchical controller improves matched-seed combat success or sample efficiency compared with Flat PPO.
- H-Risk: Risk-conditioned high-level choices reduce HP loss at comparable or greater win rate, not merely by avoiding engagement.
- H-Gate: Learned decision timing yields better performance/compute trade-off than fixed decision intervals.
- H-Jev: Optional typed advisory changes meaningful high-level decisions and improves a measured outcome net of latency/cost.

## 10. Change policy / non-negotiable invariants

1. **Only the low-level learned policy outputs native per-frame buttons/camera.** Scripted controls remain diagnostics/oracles, not the claimed Harness policy.
2. **The high level outputs a condition, not an attack macro.** It may select a tactic/embedding and decide when to reconsider.
3. **The low-level model must actually consume and learn from `z`.** A no-op parameter does not qualify.
4. **Both layers are ultimately learnable;** stage-wise training is the first implementation strategy, not a permanent claim that fixed heuristics are the target.
5. **Risk is predictive, calibrated, and observation-clean;** it is not interchangeable with the PPO value function.
6. **Jev is optional, low-frequency and fails open.** Never add mandatory external calls to every simulator step.
7. **Flat PPO remains a separately reproducible control.** Never silently change baseline inputs, rewards, seeds, or action semantics.
8. **The written architecture precedes interface-breaking changes.** When evidence warrants a revision, record an ADR/version change and required ablations rather than quietly deviating.

## 11. Immediate next task

**H0 — Hierarchical Policy Interface and Runtime**, not a rule-based Attack/Approach controller:

- Specify `HarnessState`, `TacticalDecision`, component protocols and reset semantics.
- Build a `HierarchicalCombatPolicy.get_action(...)` compatible with the existing `eval/rollout.py` calling convention.
- Use fake high-level/low-level/gate implementations and a minimal action-shaped fake environment in tests; no Minecraft, pretrained weights or GPU required.
- Verify decision update cadence, `z` propagation, action forwarding, policy state reset and error/fallback handling.
- Do **not** claim H1 (real conditioned VPT) until the policy network's input path and gradient tests exist.

## 12. References and related project documents

- Sutton, Precup & Singh (1999), [Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning](https://www.sciencedirect.com/science/article/pii/S0004370299000521).
- Bacon, Harb & Precup (2017), [The Option-Critic Architecture](https://ojs.aaai.org/index.php/AAAI/article/view/10916).
- Baker et al. (2022), [Video PreTraining: Learning to Act by Watching Unlabeled Online Videos](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c7008aff45b5d8f0973b23e1a22ada0-Abstract-Conference.html).
- TypeSafe AI, [Jev: typed decision model overview](https://www.typesafeai.org/jev) (external/advisory option, not an established part of the MineCombat stack).
- [MineCombat research plan](research-plan.md), [roadmap](roadmap.md), [M2 baselines](m2-baseline.md), [PPO smoke](m3-ppo-smoke.md), [dense-damage report](m3-dense-damage.md), [1M-step runbook](1m-run.md).
