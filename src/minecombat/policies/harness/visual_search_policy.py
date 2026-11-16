
"""Compose visual detection, closed-loop search and Flat VPT combat."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

from .search_controller import (
    ClosedLoopSearchController,
    SearchMemory,
    TargetDetection,
)


class TargetDetector(Protocol):
    def detect(self, obs: dict) -> TargetDetection:
        ...


class CombatPolicy(Protocol):
    def get_action(
        self,
        obs: dict,
        memory: Any,
        *,
        deterministic: bool,
        input_shape: str,
    ) -> tuple[dict, Any]:
        ...


class CameraActionAdapter:
    """Convert yaw degrees into MineStudio's joint agent action.

    Use the environment's own camera quantizer and action mapper.
    This ensures the camera-on meta action is set correctly.
    """

    def __init__(self, env: Any, *, yaw_sign: int = 1):
        if getattr(env, "action_type", None) != "agent":
            raise ValueError("Requires MinecraftSim action_type='agent'")

        if not callable(
            getattr(env, "env_action_to_agent_action", None)
        ):
            raise TypeError(
                "Environment must support agent-action conversion"
            )

        if type(yaw_sign) is not int or yaw_sign not in (-1, 1):
            raise ValueError("yaw_sign must be +1 or -1")

        self.env = env
        self.yaw_sign = yaw_sign

    def __call__(self, yaw_delta: float) -> dict:
        if not math.isfinite(yaw_delta):
            raise ValueError("yaw_delta must be finite")

        # Environment camera order: [pitch, yaw].
        # The input has an explicit batch dimension.
        env_action = {
            "camera": np.array(
                [[0.0, self.yaw_sign * yaw_delta]],
                dtype=np.float32,
            )
        }

        # Convert to joint discrete buttons/camera indices.
        return self.env.env_action_to_agent_action(env_action)


@dataclass(frozen=True)
class VisualSearchState:
    """Recurrent state for one Minecraft episode."""

    search_memory: SearchMemory = field(
        default_factory=SearchMemory
    )
    low_memory: Any = None

    mode: Literal["search", "align", "combat"] | None = None
    steps: int = 0

    last_detection: TargetDetection | None = None


class VisualSearchPolicy:
    """Combine visual search with an existing trained combat policy.

    SEARCH / ALIGN:
        Execute camera-only actions.

    COMBAT:
        Execute the original Flat VPT policy.

    The combat memory is reset when control leaves COMBAT,
    preventing stale recurrent state from being reused.
    """

    def __init__(
        self,
        *,
        detector: TargetDetector,
        controller: ClosedLoopSearchController,
        combat_policy: CombatPolicy,
        camera_action: Callable[[float], dict],
    ) -> None:
        self.detector = detector
        self.controller = controller
        self.combat_policy = combat_policy
        self.camera_action = camera_action

    def get_action(
        self,
        obs: dict,
        memory: VisualSearchState | None,
        *,
        deterministic: bool = True,
        input_shape: str = "*",
    ) -> tuple[dict, VisualSearchState]:
        """Run one observation-driven policy step."""

        if input_shape != "*":
            raise ValueError(
                "VisualSearchPolicy only supports single-frame input"
            )

        state = memory if memory is not None else VisualSearchState()

        if not isinstance(state, VisualSearchState):
            raise TypeError("Invalid visual search memory")

        # 1. Detect the Zombie from the RGB observation.
        detection = self.detector.detect(obs)

        # 2. Update the search / alignment / combat state.
        command, next_search_memory = self.controller.step(
            detection,
            state.search_memory,
        )

        if command.use_combat_policy:
            # 3a. Run the learned combat policy.
            # Re-entering combat starts a fresh VPT recurrent state.
            low_memory_in = (
                state.low_memory
                if state.mode == "combat"
                else None
            )

            action, next_low_memory = (
                self.combat_policy.get_action(
                    obs,
                    low_memory_in,
                    deterministic=deterministic,
                    input_shape=input_shape,
                )
            )

        else:
            # 3b. Search/align using camera-only control.
            if command.yaw_delta is None:
                raise RuntimeError(
                    "Search/align command is missing yaw_delta"
                )

            action = self.camera_action(command.yaw_delta)

            # The VPT was not executed on this frame.
            next_low_memory = None

        if (
            not isinstance(action, dict)
            or "buttons" not in action
            or "camera" not in action
        ):
            raise TypeError(
                "Expected joint agent action with buttons and camera"
            )

        # 4. Return a fresh action and the next recurrent state.
        next_state = VisualSearchState(
            search_memory=next_search_memory,
            low_memory=next_low_memory,
            mode=command.mode,
            steps=state.steps + 1,
            last_detection=detection,
        )

        return dict(action), next_state
