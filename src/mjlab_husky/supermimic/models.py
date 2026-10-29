"""Differentiable vector dynamics used by HUSKY's imagined-policy trainer.

This is an adaptation of the normalized state-delta and frozen-model BPTT
design in Breezewrf/supermimic_mjlab. Unlike its G1 tracking-specific analytical
heads, HUSKY learns task reward and physical termination from real transitions.
The simulator state/observation layout is owned by the environment adapter.
"""

from contextlib import contextmanager
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Prediction(NamedTuple):
    next_state: Tensor
    reward: Tensor
    termination_logit: Tensor

    @property
    def continuation(self) -> Tensor:
        return torch.sigmoid(-self.termination_logit)


class RunningNorm(nn.Module):
    """Parallel variance statistics, updated explicitly from real samples only."""

    def __init__(self, dimension: int, eps: float = 1e-3, max_count: int = 200_000_000):
        super().__init__()
        self.eps, self.max_count = eps, max_count
        self.register_buffer("mean", torch.zeros(dimension, dtype=torch.float64))
        self.register_buffer("variance", torch.ones(dimension, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))

    @torch.no_grad()
    def update(self, samples: Tensor) -> None:
        samples = samples.detach().reshape(-1, self.mean.numel())
        remaining = self.max_count - int(self.count)
        if not len(samples) or remaining <= 0:
            return
        samples = samples[:remaining].double()
        if not torch.isfinite(samples).all():
            raise ValueError("Normalization cannot consume nonfinite real samples")
        n = len(samples)
        mean = samples.mean(0)
        variance = samples.var(0, unbiased=False)
        old_count = self.count.to(torch.float64)
        total = old_count + n
        delta = mean - self.mean
        combined = (self.variance * old_count + variance * n
                    + delta.square() * old_count * n / total) / total
        self.mean.add_(delta * n / total)
        self.variance.copy_(combined.clamp_min(0))
        self.count.add_(n)

    def scale(self, like: Tensor) -> Tensor:
        # Before the first observation use the identity rather than tiny scales.
        variance = torch.where(self.count > 0, self.variance, torch.ones_like(self.variance))
        return variance.sqrt().clamp_min(self.eps).to(like)

    def forward(self, value: Tensor) -> Tensor:
        return (value - self.mean.to(value)) / self.scale(value)

    def decode(self, value: Tensor) -> Tensor:
        return value * self.scale(value) + self.mean.to(value)


@contextmanager
def frozen_parameters(*modules):
    """Freeze model weights while preserving derivatives through states/actions."""
    previous = [(p, p.requires_grad) for module in modules for p in module.parameters()]
    try:
        for parameter, _ in previous:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in previous:
            parameter.requires_grad_(enabled)


class WorldModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(512, 512, 512),
                 learning_rate=2e-4, device="cpu", gradient_clip=1.0,
                 project_next=None, known_reward=None, precision="float32", known_board_reward=None):
        super().__init__()
        if state_dim < 1 or action_dim < 1 or not hidden_dims or min(hidden_dims) < 1:
            raise ValueError("Model dimensions must be positive")
        if precision not in ("float32", "bfloat16"):
            raise ValueError("World-model precision must be float32 or bfloat16")
        if precision == "bfloat16":
            target_device = torch.device(device)
            if (target_device.type not in ("cpu", "cuda")
                    or not torch.amp.autocast_mode.is_autocast_available(target_device.type)):
                raise ValueError("World-model BF16 requires CPU or CUDA autocast support")
            if target_device.type == "cuda":
                with torch.cuda.device(target_device):
                    if not torch.cuda.is_bf16_supported():
                        raise ValueError("This GPU does not support BF16; use precision=float32")
        self.precision = precision
        self.state_dim, self.action_dim = state_dim, action_dim
        self.hidden_dims, self.gradient_clip = tuple(hidden_dims), gradient_clip
        # Optional task-owned constraints (e.g. exact observation-history shift).
        # A callable is configuration/code, deliberately not checkpoint state.
        self.project_next = project_next
        # Optional parameter-free task component A(s,a). The learned scalar
        # becomes R-A while Prediction.reward remains the total task reward.
        # The callable is task configuration, not a new model/module parameter.
        if known_reward is not None and (not callable(known_reward) or isinstance(known_reward, nn.Module)):
            raise ValueError("known_reward must be a parameter-free callable, not an nn.Module")
        self.known_reward = known_reward
        if known_board_reward is not None and (
                known_reward is not None or not callable(known_board_reward)
                or isinstance(known_board_reward, nn.Module)
                or not callable(getattr(known_board_reward, "metadata", None))):
            raise ValueError("Known board reward must be a parameter-free callable with metadata, used without known action costs")
        self.known_board_reward = known_board_reward
        # Training may condition known external command/noise updates on their
        # observed samples without teacher-forcing any physical prediction.
        self.project_training_next = None
        widths = (state_dim + action_dim, *hidden_dims)
        layers = []
        for left, right in zip(widths[:-1], widths[1:]):
            layers.extend((nn.Linear(left, right), nn.LayerNorm(right), nn.SiLU()))
        self.trunk = nn.Sequential(*layers)
        self.delta_head = nn.Linear(widths[-1], state_dim)
        self.reward_head = nn.Linear(widths[-1], 1)
        self.termination_head = nn.Linear(widths[-1], 1)
        # Start with the observed mean state increment, not a large random jump.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.input_norm = RunningNorm(state_dim + action_dim)
        self.delta_norm = RunningNorm(state_dim)
        self.reward_norm = RunningNorm(1, eps=1e-2)
        if precision == "bfloat16":
            # Autocast never converts master weights or Adam state to BF16.
            # Keep the existing float64 statistical accumulators unchanged.
            for network in (self.trunk, self.delta_head, self.reward_head, self.termination_head):
                network.float()
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer_updates = 0

    @torch.no_grad()
    def update_normalization(self, state: Tensor, action: Tensor, next_state: Tensor,
                             reward: Tensor, next_state_valid: Tensor | None = None, *,
                             known_s_real: Tensor | None = None):
        """Call once per newly collected transition, never for replay/imagination."""
        if self.known_board_reward is not None:
            if (known_s_real is None or known_s_real.shape != state.shape[:-1]
                    or not known_s_real.is_floating_point()
                    or not torch.isfinite(known_s_real).all()):
                raise ValueError("Board residual normalization requires every finite real known-S side label")
        elif known_s_real is not None:
            raise ValueError("Known-S labels require the board-residual representation")
        if self.precision == "bfloat16":
            state, action, next_state, reward = (value.float() for value in
                                                 (state, action, next_state, reward))
        self.input_norm.update(torch.cat((state, action), -1))
        delta = next_state - state
        if next_state_valid is not None:
            delta = delta[next_state_valid.bool()]
        self.delta_norm.update(delta)
        residual = reward
        if self.known_board_reward is not None:
            residual = reward.reshape(known_s_real.shape) - known_s_real
        if self.known_reward is not None:
            known = self.known_reward(state, action)
            if known.shape != state.shape[:-1]:
                raise ValueError("Known reward must have the state's leading batch dimensions")
            residual = reward.reshape(known.shape) - known
        # Count every real reward, including physical failures and timeouts;
        # action histories and executed action are available before any reset.
        self.reward_norm.update(residual.reshape(-1, 1))

    def forward(self, state: Tensor, action: Tensor) -> Prediction:
        if self.known_board_reward is not None:
            if self.precision == "bfloat16":
                state, action = state.float(), action.float()
            prediction = self._board_residual_prediction(state, action)
            known = self.known_board_reward(state, prediction.next_state)
            if known.shape != prediction.reward.shape:
                raise ValueError("Known board reward dimensions differ from the predicted total reward")
            return Prediction(prediction.next_state, prediction.reward + known, prediction.termination_logit)
        if self.precision == "bfloat16":
            return self._forward_bfloat16(state, action)
        # Retain the original FP32 operation order and checkpoint tensor schema.
        features = self.input_norm(torch.cat((state, action), -1))
        hidden = self.trunk(features)
        prediction = Prediction(
            state + self.delta_norm.decode(self.delta_head(hidden)),
            self.reward_norm.decode(self.reward_head(hidden)).squeeze(-1),
            self.termination_head(hidden).squeeze(-1),
        )
        if self.known_reward is None:
            return prediction
        known = self.known_reward(state, action)
        if known.shape != prediction.reward.shape:
            raise ValueError("Known reward dimensions differ from the total reward prediction")
        return Prediction(prediction.next_state, prediction.reward + known, prediction.termination_logit)

    def _board_residual_prediction(self, state: Tensor, action: Tensor) -> Prediction:
        """Raw next state, independent residual head and term logit; no S decode.

        Sequence fitting supervises this residual directly against R-S_real.
        It must not reconstruct the residual by subtracting S_pred from total R.
        Terminal dynamics stay masked; nonlinear S of the predicted conditional
        state can still have Jensen/terminal-conditioning bias in total reward.
        """
        if self.precision == "bfloat16":
            # Known action costs are mutually exclusive, so this native BF16
            # path returns the decoded residual without any analytical addend.
            return self._forward_bfloat16(state, action)
        features = self.input_norm(torch.cat((state, action), -1))
        hidden = self.trunk(features)
        return Prediction(
            state + self.delta_norm.decode(self.delta_head(hidden)),
            self.reward_norm.decode(self.reward_head(hidden)).squeeze(-1),
            self.termination_head(hidden).squeeze(-1))

    def _forward_bfloat16(self, state: Tensor, action: Tensor) -> Prediction:
        # Only MLP execution is mixed precision. Normalization, analytical task
        # terms and physical-state arithmetic stay FP32 even under outer AMP.
        with torch.autocast(device_type=state.device.type, enabled=False):
            state, action = state.float(), action.float()
            features = self.input_norm(torch.cat((state, action), -1))
            with torch.autocast(device_type=state.device.type, dtype=torch.bfloat16):
                hidden = self.trunk(features)
                delta = self.delta_head(hidden)
                reward = self.reward_head(hidden)
                termination = self.termination_head(hidden)
            prediction = Prediction(
                state + self.delta_norm.decode(delta.float()),
                self.reward_norm.decode(reward.float()).squeeze(-1),
                termination.float().squeeze(-1),
            )
            if self.known_reward is None:
                return prediction
            known = self.known_reward(state, action)
            if known.shape != prediction.reward.shape:
                raise ValueError("Known reward dimensions differ from the total reward prediction")
            return Prediction(prediction.next_state, prediction.reward + known, prediction.termination_logit)

    def sequence_loss(self, batch: dict[str, Tensor], rollout_steps: int = 8):
        """Autoregressive state/reward/termination loss; never crosses resets.

        Batch axes are [batch, time, features], scalar fields [batch, time].
        `valid` includes a terminal transition, but excludes later transitions.
        Reset next states are excluded from dynamics targets with
        `next_state_valid`; timeouts never become positive termination labels.
        """
        if self.precision == "bfloat16":
            batch = {name: value.float() if value.is_floating_point() else value
                     for name, value in batch.items()}
        steps = min(rollout_steps, batch["state"].shape[1])
        if steps < 1:
            raise ValueError("World-model rollout must contain a transition")
        state = batch["state"][:, 0].detach()
        valid = batch.get("valid", torch.ones_like(batch["reward"], dtype=torch.bool))[:, :steps]
        next_valid = batch.get("next_state_valid", ~(batch["terminated"] | batch["truncated"]))[:, :steps]
        if self.known_board_reward is not None:
            known_s_real = batch.get("known_s_real")
            if (known_s_real is None or known_s_real.shape != batch["reward"].shape
                    or not known_s_real.is_floating_point()
                    or not torch.isfinite(known_s_real).all()):
                raise ValueError("Board residual fitting requires real known-S labels for the entire sequence")
        state_sum = state.new_zeros(())
        reward_sum = state.new_zeros(())
        termination_sum = state.new_zeros(())
        for t in range(steps):
            action = batch["action"][:, t].detach()
            prediction = (self(state, action) if self.known_board_reward is None
                          else self._board_residual_prediction(state, action))
            if self.project_training_next is not None:
                next_state = self.project_training_next(
                    state, prediction.next_state, action, batch["next_state"][:, t].detach())
            else:
                next_state = (prediction.next_state if self.project_next is None
                              else self.project_next(state, prediction.next_state, action))
            state_error = ((next_state - batch["next_state"][:, t].detach())
                           / self.delta_norm.scale(state)).square().mean(-1)
            # The action-only branch is algebraically a residual error. Board
            # rewards instead use the raw head and observed R-S_real target;
            # S_pred is deliberately absent from this loss.
            reward_target = batch["reward"][:, t].detach()
            if self.known_board_reward is not None:
                reward_target = reward_target - known_s_real[:, t].detach()
            reward_error = ((prediction.reward - reward_target)
                            / self.reward_norm.scale(prediction.reward)[0]).square()
            termination_error = F.binary_cross_entropy_with_logits(
                prediction.termination_logit, batch["terminated"][:, t].float(), reduction="none")
            state_sum = state_sum + torch.where(valid[:, t] & next_valid[:, t], state_error, 0).sum()
            reward_sum = reward_sum + torch.where(valid[:, t], reward_error, 0).sum()
            termination_sum = termination_sum + torch.where(valid[:, t], termination_error, 0).sum()
            # Keep invalid padded trajectories bounded without severing valid BPTT.
            state = torch.where((valid[:, t] & next_valid[:, t])[:, None],
                                next_state, batch["next_state"][:, t].detach())
        state_loss = state_sum / (valid & next_valid).sum().clamp_min(1)
        reward_loss = reward_sum / valid.sum().clamp_min(1)
        termination_loss = termination_sum / valid.sum().clamp_min(1)
        return {"loss": state_loss + reward_loss + termination_loss,
                "state_loss": state_loss, "reward_loss": reward_loss,
                "termination_loss": termination_loss}

    def fit(self, batch: dict[str, Tensor], rollout_steps: int = 8) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.sequence_loss(batch, rollout_steps)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError("Nonfinite world-model loss")
        losses["loss"].backward()
        gradient = torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip,
                                                  error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer_updates += 1
        return {**{key: float(value.detach()) for key, value in losses.items()},
                "gradient_norm": float(gradient), "optimizer_updates": self.optimizer_updates}

    def training_state_dict(self):
        state = {"model": self.state_dict(), "optimizer": self.optimizer.state_dict(),
                "optimizer_updates": self.optimizer_updates,
                "state_dim": self.state_dim, "action_dim": self.action_dim,
                "hidden_dims": self.hidden_dims}
        if self.known_reward is not None:
            state["reward_representation"] = "action_residual"
        if self.known_board_reward is not None:
            state["reward_representation"] = "board_residual_v1"
            state["known_board_reward"] = self.known_board_reward.metadata()
        if self.precision != "float32":
            state["precision"] = self.precision
        return state

    def load_training_state_dict(self, state):
        if (state["state_dim"], state["action_dim"], tuple(state["hidden_dims"])) != (
                self.state_dim, self.action_dim, self.hidden_dims):
            raise ValueError("World-model checkpoint architecture differs")
        expected = ("board_residual_v1" if self.known_board_reward is not None else
                    "action_residual" if self.known_reward is not None else "total")
        if state.get("reward_representation", "total") != expected:
            raise ValueError("World-model reward representation differs; scalar-total and residual heads cannot be interchanged")
        if self.known_board_reward is not None and state.get("known_board_reward") != self.known_board_reward.metadata():
            raise ValueError("World-model known board reward metadata differs")
        if state.get("precision", "float32") != self.precision:
            raise ValueError("World-model training precision differs; use matching precision for optimizer resume "
                             "or explicitly load only FP32 model weights without optimizer state")
        self.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.optimizer_updates = state["optimizer_updates"]
