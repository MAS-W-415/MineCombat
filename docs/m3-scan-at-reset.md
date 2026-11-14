# M3 — Scan-at-Reset Probe (Rejected)

Date: 2026-11-14  
Checkpoint: `experiments/probe_face_100k/checkpoints/2026-11-14_13-27-50/2/`  
Config: `configs/env/zombie_stage1_random_scan.yaml`  
Protocol: 50 seeds, max 600 steps, deterministic, local RTX 4060

## Idea

The policy fails to acquire off-screen targets. A scripted 360° camera scan at
reset (36 steps × 10°) should give it a chance to see the zombie before the
policy takes over. This uses no privileged zombie position, unlike `face_spawn`.

## Result

| Task | Win | Death | Timeout | Kills |
|---|---:|---:|---:|---:|
| random (baseline) | 0.10 | 0.64 | 0.26 | 5/50 |
| random_face | **0.52** | 0.34 | 0.14 | 26/50 |
| **random_scan** | **0.00** | 0.46 | **0.54** | **0/50** |

Mean HP lost: 11.28. Mean steps: 479.1. Stuck rate: 54%.

## Interpretation

The scan made the unassisted task worse, not better:

- A full turn returns the camera to its original orientation, so a target that
  was behind the player is behind again at the end of the scan.
- The policy does not reliably re-acquire the target from its recurrent memory.
- The 36-step scan also gives the zombie time to close distance and attack; the
  death rate stays at 46%.

A fixed open-loop scan is not sufficient. A useful scan would need target
detection / stopping when the zombie is visible, which is an observation or
detector problem, not a fixed camera schedule.

## Decision

Reject scan-at-reset as implemented. Do not run more random-spawn PPO on this
mechanism.

The remaining viable directions are:

1. **Auto-face deployment / split gate**: the face-assisted task reaches 52%
   with the 100k probe policy. If the team accepts an initial-facing assist,
   Stage-1b can be defined as "random distance with auto-face" rather than full
   360° search.
2. **Relative-direction observation (HH)**: provide the agent with a bearing or
   unit vector to the zombie. This is the most direct way to make search
   learnable and belongs in the harness/observation line.
3. **Learned search**: train a separate search policy or a scan policy with a
   target-visible reward. This is a new research track, not a flat-PPO
   continuation.

## Artifacts

- `experiments/eval_probe_scan_local/20261114_143138_eval_vpt_ckpt/`
- Code: `ScanAtResetCallback`, `configs/env/zombie_stage1_random_scan.yaml`
