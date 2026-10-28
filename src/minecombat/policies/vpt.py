"""VPT policy loading."""

from __future__ import annotations

DEFAULT_MODEL = "CraftJarvis/MineStudio_VPT.foundation_model_1x"


def load_vpt_policy(repo_id: str = DEFAULT_MODEL, device: str | None = None):
    """Load a VPT policy from Hugging Face and put it in eval mode."""
    import torch
    from minestudio.models import VPTPolicy

    policy = VPTPolicy.from_pretrained(repo_id)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(device)
    policy.eval()
    return policy
