# HH1 RTX 4090D validation

Hypothesis: the conditioned VPT PPO chain should reproduce on a 24 GB GPU and
save a finite checkpoint with updated conditioner weights.

Result: the 512-step smoke and 10,240-step diagnostic completed. The final
zero-initialized projection had norm 0.123726, demonstrating a real optimizer
update. Six high-KL warnings occurred without numerical divergence.

Limitation: consumed replay fragments are deleted, so the complete per-tactic
ratio cannot be reconstructed after the run. A separate real-environment reset
probe sampled both tactic IDs.

Next action: use the preserved checkpoint for HH2 and defer behavioral claims
until a fixed-seed tactic comparison is run.
