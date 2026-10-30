"""Pickle-safe generators for the online trainer.

The rollout worker uses multiprocessing ``spawn`` and pickles the generators by
reference. Functions defined in ``__main__`` cannot be pickled this way, so they
live in an importable module.
"""

from __future__ import annotations


def env_generator(config_path: str):
    from minecombat.envs import make_zombie_env

    return make_zombie_env(config_path)


def policy_generator(repo_id: str):
    from minestudio.models import VPTPolicy
    from minestudio.utils import get_compute_device

    return VPTPolicy.from_pretrained(repo_id).to(get_compute_device())
