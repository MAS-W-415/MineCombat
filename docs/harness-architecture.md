# MineCombat: Hierarchical Combat Harness Architecture

> Version: 1.0
> Status: Design baseline — H0 implementation in progress
> Date: 2026-10-31

## 1. Project Objective

MineCombat aims to develop an intelligent Minecraft combat agent
using pretrained visual policies, reinforcement learning, and
hierarchical tactical decision-making.

The long-term objective is to progress from single-zombie combat
toward complex multi-enemy encounters and the Ender Dragon.

This document defines the intended Harness architecture.

The architecture is a research proposal, not a validated
performance improvement.

## 2. Core Design Principles

1. The low-level policy generates all native keyboard and mouse actions.
2. The high-level policy selects tactical conditions, not frame-level actions.
3. Reinforcement learning can optimize both policy levels.
4. The Harness coordinates modules without implementing scripted combat.
5. Risk prediction informs tactical decisions.
6. High-level decisions operate at a lower frequency than low-level actions.
7. External advisors are optional and must not block normal execution.
8. The flat VPT + PPO policy remains an independent experimental baseline.

## 3. Overall Architecture

```text
               Minecraft Environment
                         |
                RGB Observation
                Previous Action
                         |
                         v
              Visual / Temporal Encoder
                         |
                    Features h_t
                         |
          +--------------+--------------+
          |                             |
          v                             v
    Risk Predictor                Tactical Policy
          |                             ^
          |        Risk Estimate        |
          +---------------------------->|
                                        |
                                Tactical Condition
                                        z_t
                                        |
                                Decision Gate
                                Update / Retain
                                        |
                                        v
                              Conditioned VPT Policy
                                pi_L(a | h, z, m)
                                        |
                                        v
                                Native Actions
                             Camera / Keys / Attack
                                        |
                                        v
                                  MinecraftSim
                                        |
                                Reward / Termination
                                        |
                                        v
                                   RL Training


            Optional External Advisor (Jev)
                         |
                 Tactical Suggestions
                         |
                         v
                High-Level Arbitration
                         |
                  Tactical Policy
```

The diagram describes the logical system architecture.

The order of execution inside the Harness Runtime is:

1. Receive observation.
2. Extract features.
3. Estimate risk, if a predictor is available.
4. Evaluate the decision gate.
5. Update or retain the tactical condition.
6. Execute the low-level policy.
7. Return the native action and updated memory.

## 4. Low-Level Policy

### 4.1 Responsibility

The low-level policy directly generates Minecraft actions:

- Camera movement
- Forward/backward movement
- Strafing
- Attack and attack release
- Jump
- Sprint
- Other native agent actions

It must not be replaced by a hardcoded combat state machine.

### 4.2 Baseline

The existing baseline uses VPT 1x initialized from pretrained weights
and fine-tuned with PPO.

The flat policy is represented by:

    a_t ~ pi_L(a_t | h_t, m_t)

Where:

- a_t: native Minecraft action
- h_t: visual and temporal features
- m_t: recurrent policy memory

### 4.3 Conditioned Policy

The future hierarchical policy introduces tactical conditioning:

    a_t ~ pi_L(a_t | h_t, z_t, m_t)

Where z_t is produced by the high-level Tactical Policy.

The conditioning mechanism must be implemented inside the neural
policy. Passing z_t through a Python wrapper without influencing
the action distribution does not count as conditioned VPT.

Candidate implementations include:

- Tactical embedding injected into the policy features
- Conditional feature modulation
- Conditioning through a learned adapter

The exact injection mechanism is not finalized.

Training must verify that tactical conditions meaningfully affect
the policy's actions.

## 5. High-Level Tactical Policy

### 5.1 Responsibility

The high-level policy chooses tactical objectives or preferences.

Examples include:

- Engage
- Reposition
- Create distance
- Seek an attack opportunity

These are candidate tactical semantics, not hardcoded actions.

The high-level policy does not directly generate keyboard or
mouse commands.

### 5.2 Output

The H0 prototype represents tactical conditions as:

    TacticalDecision(
        tactic_id: int,
        confidence: float | None
    )

This discrete representation is an initial interface choice.

A learned continuous tactical embedding may replace or extend
it in later versions.

The architecture should not assume that the final representation
must be a fixed list of four tactics.

## 6. Risk Predictor

The Risk Predictor estimates future combat risk from permitted
observations and temporal features.

Candidate predictions:

- Expected HP loss over a future horizon
- Probability of player death
- Probability of a dangerous encounter

Initial interface:

    RiskEstimate(
        expected_hp_loss: float,
        death_probability: float,
        horizon_steps: int
    )

Risk prediction is distinct from the PPO value function.

The PPO value function estimates expected discounted reward.

The Risk Predictor estimates an interpretable risk target.

Privileged simulator information may be used for auxiliary
training labels when explicitly documented.

The deployed actor and risk predictor must operate from
permitted observations without privileged coordinates or
hidden enemy states.

## 7. Decision Gate

### 7.1 Purpose

The high-level policy should not be required to make a new
decision at every environment step.

The Decision Gate decides whether to:

- Retain the current tactical condition
- Request an updated tactical condition

### 7.2 H0 Baseline

H0 uses a deterministic fixed-interval gate.

Example:

    interval_steps = 20

The low-level policy still acts every frame.

The fixed-interval gate is only an engineering baseline.

### 7.3 Future Learned Gate

A future learned gate may use:

- Visual state changes
- Risk estimates
- Tactical progress
- Current tactical condition
- Historical memory

The learned gate must be evaluated against the fixed-interval
baseline rather than assumed to be superior.

## 8. Harness Runtime

The Harness Runtime coordinates the different policy modules.

It is responsible for:

- Feature extraction
- Risk prediction
- Tactical decision scheduling
- Tactical condition persistence
- Low-level policy invocation
- Episode-local state management

It is not responsible for:

- Hardcoded melee strategies
- Direct keyboard/mouse decision-making
- PPO optimizer logic
- Minecraft environment implementation

### 8.1 Episode State

The HarnessState contains:

    low_memory
    high_memory
    feature_memory
    risk_memory
    gate_memory

    current_tactic
    current_risk

    steps_since_decision
    total_steps
    last_decision_updated

The Runtime receives and returns this memory explicitly.

It must reset episode-local memory when a new evaluation
episode begins.

### 8.2 Public Interface

The Harness Runtime follows the existing evaluation contract:

```python
action, next_memory = harness.get_action(
    obs,
    memory,
    deterministic=True,
    input_shape="*",
)
```

The low-level conditioned policy additionally receives:

```python
action, next_low_memory = low_policy.get_action(
    obs,
    low_memory,
    tactical_condition=tactic,
    deterministic=True,
    input_shape="*",
)
```

H0 defines this interface.

H1 must implement the actual neural-network conditioning.

## 9. Reinforcement Learning Strategy

Training follows a progressive approach.

### Stage A: Flat VPT + PPO

Train the pretrained VPT policy directly on zombie combat.

This provides the main baseline.

The existing repository has demonstrated the PPO pipeline,
but reliable zombie combat has not yet been established.

### Stage B: Conditioned Low-Level VPT

Extend VPT to accept tactical conditions.

Train the low-level policy to respond meaningfully to
different tactical conditions.

Verify both action-distribution sensitivity and actual
behavioral differences.

### Stage C: High-Level Tactical RL

Train the Tactical Policy to choose useful conditions.

Each high-level decision persists over multiple low-level
environment steps.

High-level training must account for the duration, rewards,
and termination of each tactical segment.

Candidate approaches include hierarchical PPO and
semi-Markov decision-process formulations.

The exact optimization method remains open.

### Stage D: Risk-Aware Decision Making

Train the Risk Predictor with appropriate future-risk targets.

Evaluate whether risk prediction improves tactical decisions.

Replace or augment the fixed-interval gate with a learned gate.

### Stage E: Joint Optimization and External Advisor

Investigate joint high-level/low-level fine-tuning only after
the individual modules have been validated.

An external advisor may provide additional tactical guidance.

Any improvement must be evaluated against the same baseline
environment and fixed evaluation seeds.

## 10. External Advisor: Jev

Jev is a proposed optional high-level advisor.

Its intended responsibilities are:

- Provide low-frequency tactical suggestions
- Assist high-level strategic reasoning
- Support selected event-triggered decisions

Jev must not directly issue native Minecraft actions.

The Harness should remain functional when Jev is disabled
or unavailable.

Its API, runtime behavior, and measurable contribution
are not yet established.

## 11. Code Organization

Target package structure:

```text
src/minecombat/
    policies/
        vpt.py
        harness/
            __init__.py
            contracts.py
            gate.py
            runtime.py

            # Future modules
            tactical_policy.py
            risk_predictor.py
            conditioned_vpt.py
            advisor.py
```

Existing environment, evaluation, and PPO modules should
remain reusable.

Avoid coupling Harness-specific logic directly into
Minecraft environment callbacks.

## 12. Development Roadmap

### H0: Hierarchical Interfaces and Runtime

Deliverables:

- Tactical condition data structure
- Risk estimate data structure
- Episode-local Harness state
- High-level and low-level policy protocols
- Fixed-interval decision gate
- Harness Runtime
- Minimal integration test

Acceptance:

- High-level policy executes at configured intervals.
- Low-level policy executes at every step.
- Tactical conditions reach the low-level interface.
- Episode memory resets correctly.
- No GPU or Minecraft runtime is required.

### H1: Conditioned VPT

Deliverables:

- Concrete tactical embedding
- VPT conditioning mechanism
- Compatible recurrent inference
- Training and checkpoint support

Acceptance:

- Different conditions affect model computation.
- The conditioned network supports gradient-based learning.
- Existing flat VPT inference remains available.

### H2: High-Level Tactical RL

Deliverables:

- Trainable Tactical Policy
- Tactical transition records
- High-level reward aggregation
- Initial high-level training experiment

### H3: Risk-Aware Harness

Deliverables:

- Trained Risk Predictor
- Learned or risk-triggered Decision Gate
- Comparison against fixed-interval scheduling

### H4: External Advisor Integration

Deliverables:

- Optional external advisor interface
- Tactical arbitration mechanism
- Failure handling and fallback behavior
- Experimental comparison with and without advisor

## 13. Team Responsibilities

### Harness / Hierarchical Policy

Primary focus:

- Tactical Policy
- Tactical condition representation
- Harness Runtime
- Risk Predictor
- Decision Gate
- Conditioned Policy integration
- Hierarchical experiments

### Environment / PPO Infrastructure

Primary focus:

- Minecraft simulation and callbacks
- Reward and termination reliability
- Ray rollout infrastructure
- PPO training stability
- Checkpointing and experiment logging
- GPU experiments

### Shared Responsibilities

- Conditioning interface between Harness and VPT
- Training data and recurrent memory semantics
- Evaluation protocol
- Performance comparisons
- Final integration

Ownership should be confirmed by both contributors
before parallel changes to shared training modules.

## 14. Research Validation

The architecture must be evaluated progressively.

Required comparisons:

1. Zero-shot VPT
2. Flat VPT + PPO
3. Conditioned VPT
4. Hierarchical Policy with fixed gate
5. Risk-aware Hierarchical Policy
6. Optional external-advisor variant

Use equivalent environment configurations and fixed seeds.

Primary metric:

- Combat win rate

Secondary metrics:

- Death rate
- HP lost
- Time-to-kill
- Damage dealt
- Episode duration
- Tactical update frequency
- Inference overhead

A more complex architecture is not automatically a better one.

Each additional component must justify its computational cost
and experimental benefit.

## 15. Architectural Invariants

Future changes must preserve these requirements:

1. Only the low-level neural policy directly selects native actions.
2. High-level decisions are tactical conditions, not key sequences.
3. The flat PPO baseline remains independently executable.
4. Harness state is isolated across episodes.
5. High-level decision frequency is independently configurable.
6. Risk predictions are not treated as ground truth.
7. Privileged simulator state is not silently introduced as policy input.
8. External advisors remain optional.
9. Architectural changes require explicit documentation.
10. Unverified research hypotheses must not be reported as results.

---

This document is the architectural baseline for MineCombat's
hierarchical combat research and implementation.
