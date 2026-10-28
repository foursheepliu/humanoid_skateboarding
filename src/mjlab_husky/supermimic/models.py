"""Learned augmented robot--board dynamics, task reward and termination.

The adapter supplies an augmented coupled state: actor/critic histories, MuJoCo
robot AND skateboard state, and explicitly captured phase/command/trajectory
memory. This remains a learned approximation; omitted simulator solver, sensor
or actuator memory must not be claimed as exact Markov-state reconstruction.
No robot-only tracking model or pretrained dynamics weights are substituted.

The autoregressive fitting protocol is adapted from Breezewrf/supermimic_mjlab,
commit 77183062eb92026bfbee2daf276f87cad36fd957 (Apache-2.0). All transitions and
task labels here come from the unmodified HUSKY simulation.
"""
from collections.abc import Mapping
import inspect

import torch
from torch import nn
from torch.nn import functional as F


class RunningMoments(nn.Module):
    """Device-local real-data moments, persisted as ordinary checkpoint buffers."""

    def __init__(self, dimension: int, min_std: float = 1e-3):
        super().__init__()
        if dimension < 0 or min_std <= 0:
            raise ValueError("Invalid normalizer dimensions or minimum scale")
        self.dimension = dimension
        self.register_buffer("minimum_std", torch.tensor(float(min_std)))
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("variance", torch.ones(dimension))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @property
    def std(self):
        return torch.maximum(self.variance, self.minimum_std.square()).sqrt()

    @torch.no_grad()
    def update(self, samples):
        if samples.shape[-1] != self.dimension:
            raise ValueError("Normalizer input dimension differs")
        if self.dimension == 0 or samples.numel() == 0:
            return
        samples = samples.detach().reshape(-1, self.dimension).to(self.mean.device)
        if not bool(torch.isfinite(samples).all()):
            raise FloatingPointError("Nonfinite real data; normalizer unchanged")
        # FP64 moments reduce cancellation from world coordinates and tiny deltas.
        variance, mean = torch.var_mean(samples.double(), dim=0, unbiased=False)
        n = samples.shape[0]
        total = self.count + n
        delta = mean - self.mean.double()
        updated_mean = self.mean.double() + delta * (n / total)
        updated_variance = (
            self.variance.double() * self.count + variance * n
            + delta.square() * self.count * n / total
        ) / total
        self.mean.copy_(updated_mean.float())
        self.variance.copy_(updated_variance.clamp_min(0).float())
        self.count.copy_(total)

    def forward(self, value):
        return (value.float() - self.mean) / self.std

    def inverse(self, value):
        return value.float() * self.std + self.mean


class WorldModel(nn.Module):
    """Predict normalized full-state deltas plus actual HUSKY reward/done labels."""

    def __init__(self, state_dim, action_dim=23, context_dim=0,
                 hidden_dims=(512, 512, 512), *, mixed_precision="none"):
        super().__init__()
        if state_dim <= 0 or action_dim <= 0 or context_dim < 0 or not hidden_dims:
            raise ValueError("Positive state/action/hidden sizes are required")
        if mixed_precision not in ("none", "bf16"):
            raise ValueError("mixed_precision must be none or bf16")
        self.state_dim, self.action_dim, self.context_dim = int(state_dim), int(action_dim), int(context_dim)
        self.hidden_dims, self.mixed_precision = tuple(hidden_dims), mixed_precision
        self.state_norm = RunningMoments(state_dim)
        self.action_norm = RunningMoments(action_dim)
        self.context_norm = RunningMoments(context_dim)
        self.delta_norm = RunningMoments(state_dim)
        self.reward_norm = RunningMoments(1)
        dims = (state_dim + action_dim + context_dim, *hidden_dims)
        layers = []
        for left, right in zip(dims[:-1], dims[1:]):
            layers.extend((nn.Linear(left, right), nn.LayerNorm(right), nn.SiLU()))
        self.trunk = nn.Sequential(*layers)
        self.delta_head = nn.Linear(hidden_dims[-1], state_dim)
        self.reward_head = nn.Linear(hidden_dims[-1], 1)
        self.termination_head = nn.Linear(hidden_dims[-1], 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)
        nn.init.constant_(self.termination_head.bias, -4.0)

    def architecture(self):
        return {"state_dim": self.state_dim, "action_dim": self.action_dim,
                "context_dim": self.context_dim, "hidden_dims": self.hidden_dims,
                "mixed_precision": self.mixed_precision}

    @torch.no_grad()
    def fit_normalization(self, states, actions, contexts=None, *, next_states=None,
                          rewards=None, valid=None):
        """Update only from newly collected REAL data, never imagined transitions."""
        self.state_norm.update(states)
        self.action_norm.update(actions)
        if self.context_dim:
            if contexts is None:
                raise ValueError("This model requires its physical randomization context")
            self.context_norm.update(contexts)
        if next_states is not None:
            delta = next_states - states
            if valid is not None:
                delta = delta[valid.bool()]
            self.delta_norm.update(delta)
        if rewards is not None:
            self.reward_norm.update(rewards.reshape(-1, 1))

    def forward(self, state, action, context=None):
        if state.shape[-1] != self.state_dim or action.shape[-1] != self.action_dim:
            raise ValueError("World model state/action dimensions differ")
        parts = [self.state_norm(state), self.action_norm(action)]
        if self.context_dim:
            if context is None:
                raise ValueError("Missing physical randomization context")
            parts.append(self.context_norm(context))
        inputs = torch.cat(parts, dim=-1)
        if self.mixed_precision == "bf16" and inputs.is_cuda:
            # Only MLP operations are reduced precision. Physical states, reward,
            # termination probabilities and all losses stay FP32.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = self.trunk(inputs)
                delta, reward, logit = self.delta_head(hidden), self.reward_head(hidden), self.termination_head(hidden)
            delta, reward, logit = delta.float(), reward.float(), logit.float()
        else:
            hidden = self.trunk(inputs)
            delta, reward, logit = self.delta_head(hidden), self.reward_head(hidden), self.termination_head(hidden)
        return {"next_state": state.float() + self.delta_norm.inverse(delta),
                "reward": self.reward_norm.inverse(reward).squeeze(-1),
                "normalized_reward": reward.squeeze(-1),
                "termination_logit": logit.squeeze(-1),
                "termination_probability": torch.sigmoid(logit.squeeze(-1))}


def _field(batch, *names, default=None):
    for name in names:
        if isinstance(batch, Mapping) and name in batch:
            return batch[name]
        if hasattr(batch, name):
            return getattr(batch, name)
    return default


def detach_exogenous(value):
    """Keep externally supplied random commands outside the differentiable graph."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, Mapping):
        return {name: detach_exogenous(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_exogenous(item) for item in value)
    if isinstance(value, list):
        return [detach_exogenous(item) for item in value]
    return value


def projection_function(project):
    """Adapt two/three-argument projections and keyword-only exogenous inputs.

    The returned callable always accepts (previous, predicted, action,
    exogenous=None). Older projections ignore the optional exogenous value.
    """
    if project is None:
        return lambda previous, predicted, action, exogenous=None: predicted
    parameters = tuple(inspect.signature(project).parameters.values())
    uses_action = any(p.kind == p.VAR_POSITIONAL for p in parameters) or sum(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters
    ) >= 3
    uses_exogenous = any(
        p.kind == p.VAR_KEYWORD or
        (p.name == "exogenous" and p.kind != p.POSITIONAL_ONLY)
        for p in parameters
    )
    if uses_action and uses_exogenous:
        return lambda previous, predicted, action, exogenous=None: project(
            previous, predicted, action, exogenous=exogenous)
    if uses_exogenous:
        return lambda previous, predicted, action, exogenous=None: project(
            previous, predicted, exogenous=exogenous)
    if uses_action:
        return lambda previous, predicted, action, exogenous=None: project(previous, predicted, action)
    return lambda previous, predicted, action, exogenous=None: project(previous, predicted)


def world_model_loss(model, batch, *, project=None, reward_weight=1.0,
                     termination_weight=1.0, horizon=8, exogenous_from_state=None):
    """Eight-step autoregressive dynamics/reward/physical-failure fitting.

    `valid` gates NEXT-STATE regression (terminal post-reset data is unavailable).
    Reward/done on the terminal transition are still trained. Every loss after a
    real terminal is masked out, even if a caller supplies a padded window.
    Time-limit truncations must be excluded from the physical `dones` labels by
    the environment adapter, or supplied separately as `terminated`.

    An optional adapter extractor supplies ONLY recorded external commands to
    projection. Physical quantities must not be included: robot/board dynamics
    remain autoregressive and are compared against independent real targets.
    """
    states = _field(batch, "states", "state")
    actions = _field(batch, "actions", "action")
    rewards = _field(batch, "rewards", "reward")
    dones = _field(batch, "dones", "done", "terminated")
    terminated = _field(batch, "terminated", "physical_dones", default=dones)
    if terminated is None:
        terminated = dones
    valid = _field(batch, "valid", "next_valid")
    context = _field(batch, "context", "contexts")
    transition_valid = _field(batch, "transition_valid", "input_valid")
    if states is None or actions is None or rewards is None or dones is None:
        raise ValueError("World-model batch needs states/actions/rewards/physical dones")
    length = min(int(horizon), actions.shape[1])
    if length < 1 or states.shape[1] < length + 1:
        raise ValueError("Autoregressive windows need at least H+1 states")
    if valid is None:
        valid = torch.ones_like(dones, dtype=torch.bool)
    if transition_valid is None:
        transition_valid = torch.ones_like(dones, dtype=torch.bool)
    project_next = projection_function(project)
    state = states[:, 0].detach()
    active = torch.ones_like(dones[:, 0], dtype=torch.bool)
    state_error = state.new_zeros(())
    reward_error = state.new_zeros(())
    done_error = state.new_zeros(())
    state_count = state.new_zeros(())
    transition_count = state.new_zeros(())
    for step in range(length):
        context_t = context[:, step] if context is not None and context.ndim == 3 else context
        prediction = model(state, actions[:, step], context_t)
        exogenous = None
        if exogenous_from_state is not None:
            with torch.no_grad():
                exogenous = detach_exogenous(exogenous_from_state(states[:, step + 1].detach()))
        next_state = project_next(state, prediction["next_state"], actions[:, step], exogenous)
        input_mask = active & transition_valid[:, step].bool()
        state_mask = input_mask & valid[:, step].bool()
        normalized_error = (next_state - states[:, step + 1]) / model.delta_norm.std
        normalized_error = torch.where(state_mask[:, None], normalized_error, torch.zeros_like(normalized_error))
        state_error = state_error + normalized_error.square().mean(-1).sum()
        target_reward = model.reward_norm(rewards[:, step, None]).squeeze(-1)
        reward_delta = torch.where(input_mask, prediction["normalized_reward"] - target_reward, 0.0)
        reward_error = reward_error + reward_delta.square().sum()
        labels = torch.where(input_mask, terminated[:, step].float(), 0.0)
        logits = torch.where(input_mask, prediction["termination_logit"], 0.0)
        done_error = done_error + (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * input_mask).sum()
        state_count = state_count + state_mask.float().sum()
        transition_count = transition_count + input_mask.float().sum()
        active = input_mask & ~dones[:, step].bool()
        # Do not propagate a terminal/reset discontinuity through dynamics. This
        # remains fully autoregressive throughout every valid real episode.
        state = torch.where(active[:, None], next_state, states[:, step + 1].detach())
    state_error = state_error / state_count.clamp_min(1)
    reward_error = reward_error / transition_count.clamp_min(1)
    done_error = done_error / transition_count.clamp_min(1)
    total = state_error + reward_weight * reward_error + termination_weight * done_error
    return {"loss": total, "state": state_error, "reward": reward_error, "termination": done_error,
                   "state_transitions": state_count.detach(), "labeled_transitions": transition_count.detach()}
