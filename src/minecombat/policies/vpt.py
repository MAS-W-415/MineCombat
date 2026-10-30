"""VPT policy loading utilities."""

from __future__ import annotations

DEFAULT_MODEL = "CraftJarvis/MineStudio_VPT.foundation_model_1x"


def load_vpt_policy(
    repo_id: str = DEFAULT_MODEL,
    device: str | None = None,
):
    """Load a pretrained VPT policy for inference.

    Parameters
    ----------
    repo_id:
        Hugging Face model repository.
    device:
        PyTorch device, such as "cpu" or "cuda".
        None or "auto" selects CUDA when available,
        otherwise CPU.
    """

    # Lazy imports: only load heavy dependencies when needed.
    import torch
    from minestudio.models import VPTPolicy

    # Resolve the inference device.
    if device is None or device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # Fail before downloading weights if CUDA is requested
    # but no compatible GPU is available.
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() "
            "is False."
        )

    # Construct the model and load pretrained parameters.
    policy = VPTPolicy.from_pretrained(repo_id)

    # Move model parameters to the selected device.
    policy = policy.to(device)

    # Disable training-specific behavior during evaluation.
    policy.eval()

    return policy