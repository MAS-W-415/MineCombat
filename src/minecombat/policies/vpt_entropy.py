"""Camera-weighted entropy for VPT action heads.

The VPT action head contains two categorical distributions: ``buttons`` (8641
actions) and ``camera`` (121 actions). A global entropy bonus is dominated by
the much larger button distribution, so it gives the camera head very little
exploration pressure. This wrapper keeps the checkpoint layout unchanged and
scales only the camera entropy term.
"""

from __future__ import annotations

from typing import Any

from minestudio.utils.vpt_lib.action_head import DictActionHead


class CameraWeightedActionHead(DictActionHead):
    """A ``DictActionHead`` with a separate scale for camera entropy."""

    def __init__(
        self,
        heads: DictActionHead,
        camera_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.update(heads)
        self.camera_scale = float(camera_scale)
        if self.camera_scale <= 0:
            raise ValueError("camera_scale must be positive")

    def entropy(
        self,
        logits: dict[str, Any],
        return_dict: bool = False,
    ):
        parts = super().entropy(logits, return_dict=True)
        if return_dict:
            return parts
        return parts["buttons"] + self.camera_scale * parts["camera"]


def apply_camera_entropy_scale(
    policy,
    camera_scale: float,
):
    """Replace a policy's action head with the camera-weighted variant."""

    if camera_scale == 1.0:
        return policy
    policy.pi_head = CameraWeightedActionHead(
        policy.pi_head,
        camera_scale=camera_scale,
    )
    return policy
