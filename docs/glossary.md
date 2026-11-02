# Glossary and Naming Rules

This project used the labels `H1` / `H2` for two different things. To avoid
ambiguity, use the names below in all new issues, pull requests, documents, and
experiment reports.

## Milestones

| Name | Meaning |
|---|---|
| `M0` | Environment bring-up |
| `M1` | Zombie arena |
| `M2` | Baselines and metrics |
| `M3` | First PPO run and stage-1 gate |
| `M4` | Curriculum: skeleton, creeper, dragon |

Milestones are project phases, not research hypotheses.

## Research questions: RQ1–RQ3

These are the scientific questions in `docs/research-plan.md`.

| Name | Question | Status |
|---|---|---|
| `RQ1` | Can flat VPT + PPO learn zombie combat? | In progress |
| `RQ2` | Does zombie combat transfer to skeletons? | Not started |
| `RQ3` | Does a high-level harness improve long-horizon boss performance? | Not started |

Use `RQ1`, `RQ2`, `RQ3`. Do not write bare `H1`, `H2`, `H3` for these.

## Harness stages: HH0–HH4

These are implementation stages in `docs/harness-architecture.md`.

| Name | Stage | Status |
|---|---|---|
| `HH0` | Harness interfaces and runtime | Implemented |
| `HH1` | Conditioned VPT with tactical IDs | 10k validation complete; behavior weak |
| `HH2` | High-level tactical PPO | Implemented; GPU validation pending |
| `HH3` | Risk-aware harness | Not started |
| `HH4` | External advisor (Jev) | Not started |

Use `HH1`, `HH2`, etc. Do not write bare `H1`, `H2` for these.

## Flat policy vs harness policy

| Term | Meaning |
|---|---|
| Flat PPO | VPT 1x fine-tuned directly on the combat task; the control baseline |
| Conditioned VPT | VPT with a tactic-ID conditioner (HH1) |
| High-level PPO | A separate policy that chooses tactic IDs for a frozen conditioned VPT (HH2) |

## Historical filenames

Some old paths still contain the ambiguous names. They are kept for
reproducibility; do not treat them as the new naming convention.

| Old path or label | New label |
|---|---|
| `docs/h1-tactical-training-runbook.md` | `docs/hh1-tactical-training-runbook.md` |
| `docs/h2-online-training-runbook.md` | `docs/hh2-online-training-runbook.md` |
| `docs/h1-validation.md` | `docs/hh1-validation.md` |
| `experiments/h1_tactical_*` | historical HH1 artifacts |
| `experiments/.../eval_conditioned_tactic_*` | HH1 tactic evaluation artifacts |
| `H1` in old code comments | `HH1` |
| `H2` in old code comments | `HH2` |
| `--h1-checkpoint` CLI flag | `--hh1-checkpoint` (legacy alias still accepted) |

Code module names such as `high_level.py`, `high_level_factory.py`, and
`ConditionedVPTPolicy` are stable implementation names. They implement HH2 and
HH1 respectively; they are not part of the naming conflict.

## Rule of thumb

If the sentence is about a **research question**, write `RQ`.
If the sentence is about a **harness implementation stage**, write `HH`.
If the sentence is about a **project phase**, write `M`.

Never use a bare `H1` or `H2` in new writing.
