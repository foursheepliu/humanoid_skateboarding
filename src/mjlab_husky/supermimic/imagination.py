"""Actor BPTT through frozen learned coupled dynamics, never real MuJoCo.

Adapted training protocol: Breezewrf/supermimic_mjlab
77183062eb92026bfbee2daf276f87cad36fd957 (Apache-2.0).
"""
from contextlib import contextmanager

import torch

from .models import projection_function


@contextmanager
def frozen_parameters(*modules):
    """Freeze weights without disabling differentiable state/action Jacobians."""
    parameters = {}
    for module in modules:
        for parameter in module.parameters():
            parameters[id(parameter)] = (parameter, parameter.requires_grad)
    try:
        for parameter, _ in parameters.values():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in parameters.values():
            parameter.requires_grad_(requires_grad)


@torch.no_grad()
def gae_targets(rewards, continuation, values, gamma=0.99, lam=0.95, valid=None):
    """Detached lambda returns; neither physical terminals nor padding leak."""
    if values.shape[1] != rewards.shape[1] + 1:
        raise ValueError("GAE needs H+1 values for H rewards")
    if valid is None:
        valid = torch.ones_like(rewards, dtype=torch.bool)
    advantage = torch.zeros_like(rewards[:, 0])
    targets = []
    for step in reversed(range(rewards.shape[1])):
        mask = valid[:, step].bool()
        cont = continuation[:, step] * mask
        delta = rewards[:, step] + gamma * cont * values[:, step + 1] - values[:, step]
        advantage = torch.where(mask, delta + gamma * lam * cont * advantage, 0.0)
        targets.append(torch.where(mask, advantage + values[:, step], 0.0))
    return torch.stack(list(reversed(targets)), dim=1).detach()


def critic_loss(values, targets, mask=None):
    """Sum valid temporal squared errors, then average complete trajectories."""
    if mask is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    error = torch.where(mask.bool(), values - targets.detach(), 0.0)
    return error.double().square().sum(dim=1).mean().to(values.dtype)


def imagine(actor_critic, world, starts, state_to_observations, *, horizon=32,
            gamma=0.99, lam=0.95, context=None, project=None,
            reward_callback=None, additional_frozen_modules=()):
    """Return a genuine learned-model actor objective and detached critic data.

    The original HUSKY ActorCritic is reused for compatible actor/normalizer
    export. `project` may enforce term-major actor-history shifts, quaternion
    normalization and externally known command/phase transitions. The optional
    adversarial-style callback receives
      (state, action, next_state, predicted_task_reward, step_index)
    and returns total reward. Its discriminator should appear in
    `additional_frozen_modules` so its input gradient reaches the actor.
    """
    if horizon < 1:
        raise ValueError("Imagined horizon must be positive")
    if isinstance(starts, dict):
        state = starts.get("state", starts.get("states"))
        if context is None:
            context = starts.get("context")
    else:
        state = starts
    if state is None or state.ndim != 2:
        raise ValueError("Imagined starts must be [batch, full augmented state]")
    project_next = projection_function(project)
    state = state.detach()
    if context is not None:
        context = context.detach()
    states, actions, rewards, continuations = [state], [], [], []
    # Critic weights are frozen here but the final V(s_H) keeps dV/ds_H.
    with frozen_parameters(world, actor_critic.critic, *additional_frozen_modules):
        for step in range(horizon):
            observation = state_to_observations(state)
            action = actor_critic.act_inference(observation)
            context_t = context[:, step] if context is not None and context.ndim == 3 else context
            prediction = world(state, action, context_t)
            next_state = project_next(state, prediction["next_state"], action)
            reward = prediction["reward"]
            if reward_callback is not None:
                reward = reward_callback(state, action, next_state, reward, step)
            reward = reward.squeeze(-1) if reward.ndim > 1 else reward
            continuation = 1.0 - prediction["termination_probability"]
            actions.append(action)
            rewards.append(reward)
            continuations.append(continuation)
            states.append(next_state)
            state = next_state
        rewards = torch.stack(rewards, dim=1)
        continuation = torch.stack(continuations, dim=1)
        alive = continuation.cumprod(dim=1)
        before = torch.cat((torch.ones_like(alive[:, :1]), alive[:, :-1]), dim=1)
        discount = gamma ** torch.arange(horizon, device=state.device, dtype=state.dtype)
        reward_objective = (discount * before * rewards).sum(dim=1)
        terminal_value = actor_critic.evaluate(state_to_observations(state)).squeeze(-1)
        bootstrap_objective = gamma**horizon * alive[:, -1] * terminal_value
        objective = (reward_objective + bootstrap_objective).mean()

    # Only detached imagined observations and targets train the critic. No actor
    # graph or world-model gradient is retained by these tensors.
    with torch.no_grad():
        stacked_states = torch.stack([value.detach() for value in states], dim=1)
        batch, length, features = stacked_states.shape
        all_observations = state_to_observations(stacked_states.reshape(batch * length, features))
        values = actor_critic.evaluate(all_observations).reshape(batch, length)
        targets = gae_targets(rewards.detach(), continuation.detach(), values, gamma, lam)
        observations = {name: value.reshape(batch, length, -1)[:, :-1].reshape(batch * horizon, -1).detach()
                        for name, value in all_observations.items()}
    return {"objective": objective, "actor_loss": -objective,
            "reward_objective": reward_objective.mean(),
            "bootstrap_objective": bootstrap_objective.mean(),
            "critic_observations": observations, "targets": targets,
            "rewards": rewards.detach(), "continuation": continuation.detach(),
            "values": values.detach(), "mask": torch.ones_like(targets, dtype=torch.bool),
            "survival": alive[:, -1].detach(), "states": stacked_states,
            "actions": torch.stack(actions, dim=1).detach()}
