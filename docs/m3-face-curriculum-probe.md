# M3 — Face-Curriculum Probe (100k)

Date: 2026-11-14  
Source: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/`  
Training config: `configs/env/zombie_stage1_random_face.yaml`  
Checkpoint: `experiments/probe_face_100k/checkpoints/2026-11-14_13-27-50/2/`  
Hardware: local RTX 4060, 31 GB RAM

## Why

The face-spawn diagnostic showed that facing the zombie at reset raises random
spawn win from 10% to 36%, and episodes with damage from 16% to 52%. The next
question: can 100k of training on the face-assisted random-distance task teach
tracking/combat, and does it transfer back to the unassisted random task?

## Training

```text
steps: 102,400
iterations: 2
wall time: 1,756.37 s (29.3 min)
checkpoints: 1 and 2 retained
NaN / OOM: none
```

## Evaluation (50 seeds each, local)

| Task | 2M baseline | 2M + 100k face | Delta |
|---|---:|---:|---:|
| random_face | 0.36 | **0.52** | +0.16 |
| random (no face) | 0.10 | 0.10 | 0.00 |
| combined 3–7 + ±30° | 0.62 (remote 3×50) | 0.68 (local 50) | +0.06* |

\* The baseline combined number is remote 3×50; the probe combined number is a
local single 50. Host/renderer and sample size differ, so treat the combined
delta as directional only.

## Interpretation

1. **Face-assisted training works on its own distribution.** `random_face`
   rises from 36% to 52%; the policy learns to fight targets at random distance
   when it starts facing them.
2. **There is zero transfer to unassisted random.** `random` stays at 10%.
   Training with a face assist does not teach the policy to acquire an
   off-screen target.
3. **Search must be handled explicitly.** Stage-1b is not solved by more combat
   training; it needs a search mechanism (scan-at-reset, auto-face deployment,
   or a relative-direction observation).

## Next

1. Implement scan-at-reset and evaluate the 100k probe checkpoint with/without
   the scan on the same 50 seeds.
2. If scan raises the unassisted random win rate, combine scan with the
   face-trained checkpoint and consider a short curriculum.
3. If scan does not help, the missing piece is an observation/assist (e.g.
   relative direction) rather than more PPO steps.

## Artifacts

- Training: `experiments/probe_face_100k/`
- Evaluation: `experiments/eval_probe_face_100k/`
- Baseline face diagnostic: `docs/m3-face-spawn-diagnostic.md`
