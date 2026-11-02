"""VPT policy loading utilities."""

from __future__ import annotations

DEFAULT_MODEL = "CraftJarvis/MineStudio_VPT.foundation_model_1x"


def _resolve_device(device: str | None):
    import torch

    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() "
            "is False."
        )

    return device


def load_vpt_model(
    repo_id: str = DEFAULT_MODEL,
    device: str | None = None,
    checkpoint: str | None = None,
    *,
    eval_mode: bool = False,
    camera_entropy_scale: float = 1.0,
):
    """Load VPT weights, optionally warm-started from a checkpoint.

    Shared by evaluation and training. ``eval_mode=True`` disables training
    behavior; ``eval_mode=False`` leaves the model in training mode.
    ``camera_entropy_scale`` only affects the entropy used by PPO; it leaves
    the checkpoint layout unchanged.
    """

    import torch
    from minestudio.models import VPTPolicy

    policy = VPTPolicy.from_pretrained(repo_id)

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        policy.load_state_dict(state, strict=True)

    policy = policy.to(_resolve_device(device))

    if camera_entropy_scale != 1.0:
        from .vpt_entropy import apply_camera_entropy_scale

        policy = apply_camera_entropy_scale(
            policy,
            camera_entropy_scale,
        )

    if eval_mode:
        policy.eval()

    return policy


def load_vpt_policy(
    repo_id: str = DEFAULT_MODEL,
    device: str | None = None,
    checkpoint: str | None = None,
):
    """Load a pretrained VPT policy for inference."""

    return load_vpt_model(
        repo_id=repo_id,
        device=device,
        checkpoint=checkpoint,
        eval_mode=True,
    )
