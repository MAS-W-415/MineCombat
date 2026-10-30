"""Runtime compatibility patches for MineStudio.

MineStudio 1.1.6's ``NormalizeEwma`` adds an extra singleton dimension when it
normalizes a 1-D value target: a ``(1, 64)`` target becomes ``(1, 1, 64)``.
PyTorch broadcasts it, so the MSE is numerically the same, but the warning can
hide real shape bugs. Reshape only when the element count is identical; leave
genuinely mismatched targets untouched so the error surfaces.
"""

from __future__ import annotations


def patch_scaled_mse_head() -> None:
    """Make ``ScaledMSEHead.loss`` compare tensors with the same shape."""

    import torch.nn.functional as F
    from minestudio.utils.vpt_lib.scaled_mse_head import ScaledMSEHead

    if getattr(ScaledMSEHead, "_minecombat_patched", False):
        return

    def loss(self, prediction, target, reduction="mean"):
        normalized = self.normalizer(target)
        if (
            normalized.shape != prediction.shape
            and normalized.numel() == prediction.numel()
        ):
            normalized = normalized.reshape(prediction.shape)
        return F.mse_loss(prediction, normalized, reduction=reduction)

    ScaledMSEHead.loss = loss
    ScaledMSEHead._minecombat_patched = True
