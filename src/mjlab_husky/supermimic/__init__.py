"""Learned coupled robot--skateboard dynamics and imagined actor training.

This backend is model based. Its updates are not PPO updates and its prediction
gradients do not pass through the real MuJoCo simulator.
"""

from .models import RunningMoments, WorldModel, world_model_loss
from .imagination import frozen_parameters, gae_targets, imagine, critic_loss

__all__ = [
    "RunningMoments", "WorldModel", "world_model_loss", "frozen_parameters",
    "gae_targets", "imagine", "critic_loss",
]
