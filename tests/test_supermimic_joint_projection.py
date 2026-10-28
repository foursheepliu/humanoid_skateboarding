"""CPU identities/Jacobians for opt-in robot joint consistency; no simulator."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

from test_supermimic_replay import CoupledStateAdapter, FakeEnvironment, _adapter


class JointEnvironment(FakeEnvironment):
    def __init__(self):
        super().__init__(2)
        self.sim.data.qpos = torch.randn(2, 44)*.1
        self.sim.data.qpos[:, 3:7] = torch.tensor([1., 0., 0., 0.])
        self.sim.data.qpos[:, 33:37] = torch.tensor([1., 0., 0., 0.])
        self.sim.data.qvel = torch.randn(2, 42)*.1
        self.sim.mj_model.jnt_type = [0]+[3]*23+[0]+[3]*7
        self.sim.mj_model.jnt_qposadr = [0]+list(range(7, 30))+[30]+list(range(37, 44))
        self.sim.mj_model.joint = lambda i: SimpleNamespace(name=
            "skateboard/base" if i == 24 else "robot/base" if i == 0 else f"robot/joint_{i-1}")
        self.robot.joint_names = [f"joint_{i}" for i in range(23)]
        self.robot.data.indexing = SimpleNamespace(joint_q_adr=torch.arange(7, 30), joint_v_adr=torch.arange(6, 29))
        self.robot.data.joint_pos = self.sim.data.qpos[:, 7:30]
        self.robot.data.default_joint_pos = torch.linspace(-.2, .2, 23).repeat(2, 1)
        self.robot.data.default_joint_vel = torch.zeros(2, 23)
        self.obs_buf = {"policy": torch.randn(2, 230), "critic": torch.cat((
            self.robot.data.joint_pos-self.robot.data.default_joint_pos, self.sim.data.qvel[:, 6:29]*.05), -1)}
        manager = self.observation_manager
        manager._group_obs_term_names = {"policy": ["joint_pos", "joint_vel"], "critic": ["joint_pos", "joint_vel"]}
        manager._group_obs_term_dim = {"policy": [(115,), (115,)], "critic": [(23,), (23,)]}
        manager._group_obs_term_cfgs = {
            "policy": [SimpleNamespace(history_length=5, scale=None), SimpleNamespace(history_length=5, scale=.05)],
            "critic": [SimpleNamespace(history_length=0, scale=None), SimpleNamespace(history_length=0, scale=.05)]}


class JointProjectionTests(unittest.TestCase):
    def adapter(self, enabled=False):
        adapter = CoupledStateAdapter(JointEnvironment(), joint_consistency=enabled)
        self.addCleanup(adapter.close)
        return adapter

    def test_live_metadata_uses_indexing_defaults_and_no_new_state_fields(self):
        off, on = self.adapter(), self.adapter(True)
        self.assertEqual(off.state_dim, on.state_dim)
        self.assertEqual(off.context_dim, on.context_dim)
        self.assertEqual(off.layout, on.layout)
        self.assertFalse(off.joint_consistency)
        self.assertEqual(off.joint_projection_variant, "none")
        spec = on.joint_projection_metadata["spec"]
        self.assertEqual(spec["qpos_addresses"], list(range(7, 30)))
        self.assertEqual(spec["qvel_addresses"], list(range(6, 29)))
        self.assertEqual(on.joint_projection_metadata["spec_sha256"], _adapter.joint_projection_spec_sha256(spec))
        self.assertEqual(spec, off.joint_projection_metadata["spec"])

    def test_default_off_matches_pre_flag_projection_bit_for_bit(self):
        adapter = self.adapter()
        previous = adapter.observe()
        predicted = torch.randn_like(previous)
        off = adapter.project(previous, predicted, torch.ones(2, 3))
        del adapter.joint_consistency
        original = adapter.project(previous, predicted, torch.ones(2, 3))
        self.assertTrue(torch.equal(off, original))

    def test_exact_joints_amp_history_and_unchanged_noisy_actor(self):
        adapter = self.adapter()
        previous = adapter.observe()
        predicted = torch.randn_like(previous)
        baseline = adapter.project(previous, predicted, torch.ones(2, 3))
        adapter.set_joint_projection_metadata(adapter.joint_projection_metadata["spec"])
        result = adapter.project(previous, predicted, torch.ones(2, 3))
        spec = adapter.joint_projection_metadata["spec"]
        q = predicted[:, adapter.layout["qpos"]][:, spec["qpos_addresses"]]
        v = predicted[:, adapter.layout["qvel"]][:, spec["qvel_addresses"]]
        self.assertTrue(torch.equal(result[:, adapter.layout["policy"]], baseline[:, adapter.layout["policy"]]))
        self.assertTrue(torch.equal(result[:, adapter.layout["qpos"]], baseline[:, adapter.layout["qpos"]]))
        self.assertTrue(torch.equal(result[:, adapter.layout["qvel"]], baseline[:, adapter.layout["qvel"]]))
        torch.testing.assert_close(result[:, adapter.layout["amp_joints"]], q, rtol=0, atol=0)
        torch.testing.assert_close(result[:, slice(*spec["critic_joint_pos_slice"])],
                                   q-torch.tensor(spec["default_joint_pos"]), rtol=0, atol=0)
        torch.testing.assert_close(result[:, slice(*spec["critic_joint_vel_slice"])], v*.05, rtol=0, atol=0)
        history = result[:, adapter.layout["amp_history"]].reshape(2, 5, 23)
        old = previous[:, adapter.layout["amp_history"]].reshape(2, 5, 23)
        torch.testing.assert_close(history[:, :-1], old[:, 1:], rtol=0, atol=0)
        torch.testing.assert_close(history[:, -1], q, rtol=0, atol=0)

    def test_joint_gradients_route_through_physics_not_independent_heads(self):
        adapter = self.adapter(True)
        previous = adapter.observe()
        with torch.inference_mode():
            adapter.project(previous, previous)
        self.assertTrue(all(not torch.is_inference(value) and not value.requires_grad
                            for cache in adapter._joint_projection_cache.values() for value in cache.values()))
        predicted = torch.randn_like(previous).requires_grad_(True)
        result = adapter.project(previous, predicted)
        spec = adapter.joint_projection_metadata["spec"]
        loss = (result[:, adapter.layout["amp_joints"]].sum()
            + result[:, adapter.layout["amp_history"]].reshape(2, 5, 23)[:, -1].sum()
            + result[:, slice(*spec["critic_joint_pos_slice"])].sum()
            + result[:, slice(*spec["critic_joint_vel_slice"])].sum())
        loss.backward()
        torch.testing.assert_close(predicted.grad[:, adapter.layout["qpos"]][:, 7:30], torch.full((2, 23), 3.))
        torch.testing.assert_close(predicted.grad[:, adapter.layout["qvel"]][:, 6:29], torch.full((2, 23), .05))
        self.assertEqual(float(predicted.grad[:, adapter.layout["amp_joints"]].abs().sum()), 0.)
        self.assertEqual(float(predicted.grad[:, adapter.layout["critic"]].abs().sum()), 0.)

    def test_inference_warm_cache_then_real_world_loss_and_actor_bptt_backward(self):
        from mjlab_husky.supermimic.models import WorldModel, world_model_loss
        from mjlab_husky.supermimic.imagination import imagine
        adapter = self.adapter(True)
        state = adapter.observe().detach()
        with torch.inference_mode():
            adapter.project(state, state)
        world = WorldModel(adapter.state_dim, 3, hidden_dims=(16,))
        with torch.no_grad():
            torch.nn.init.normal_(world.delta_head.weight, std=.01)
            torch.nn.init.normal_(world.reward_head.weight, std=.01)
        horizon = 4
        batch = {"states": state[:, None].repeat(1, horizon+1, 1),
            "actions": torch.randn(2, horizon, 3)*.1, "rewards": torch.zeros(2, horizon),
            "dones": torch.zeros(2, horizon, dtype=torch.bool),
            "valid": torch.ones(2, horizon, dtype=torch.bool)}
        losses = world_model_loss(world, batch, project=adapter.project,
            exogenous_from_state=adapter.exogenous_from_state, horizon=horizon)
        losses["loss"].backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in world.parameters() if p.grad is not None), 0.)
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in world.parameters()))
        world.zero_grad(set_to_none=True)

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = torch.nn.Linear(adapter.policy_dim, 3)
                self.critic = torch.nn.Linear(adapter.critic_dim, 1)
            def act_inference(self, obs): return self.actor(obs["policy"])
            def evaluate(self, obs): return self.critic(obs["critic"])

        policy = Policy()
        imagined = imagine(policy, world, state, adapter.state_to_observations,
            horizon=8, project=adapter.project, exogenous_sampler=adapter.sample_exogenous)
        imagined["actor_loss"].backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in policy.actor.parameters()), 0.)
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in policy.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in world.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.critic.parameters()))

    def test_eight_step_actor_bptt_with_world_weights_frozen(self):
        adapter = self.adapter(True)
        actor = torch.nn.Linear(adapter.policy_dim, 3)
        world = torch.nn.Linear(adapter.state_dim+3, adapter.state_dim).requires_grad_(False)
        state = adapter.observe().detach()
        loss = torch.zeros(())
        for _ in range(8):
            action = actor(state[:, adapter.layout["policy"]])
            state = adapter.project(state, world(torch.cat((state, action), -1)), action)
            loss = loss+state[:, adapter.layout["amp_joints"]].square().mean()
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in actor.parameters()), 0.)
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in actor.parameters()))
        self.assertTrue(all(p.grad is None for p in world.parameters()))

    def test_invalid_duplicate_addresses_scales_or_varying_defaults_rejected(self):
        adapter = self.adapter()
        spec = adapter.joint_projection_metadata["spec"]
        bad = copy.deepcopy(spec); bad["qpos_addresses"][1] = bad["qpos_addresses"][0]
        with self.assertRaises(ValueError): adapter.set_joint_projection_metadata(bad)
        bad = copy.deepcopy(spec); bad["critic_joint_vel_scale"][0] = 1.
        with self.assertRaises(ValueError): adapter.set_joint_projection_metadata(bad)
        env = JointEnvironment(); env.robot.data.default_joint_pos[1, 0] += .1
        with self.assertRaises(ValueError): CoupledStateAdapter(env, joint_consistency=True)

    def test_snapshot_metadata_never_enables_offline_projection_implicitly(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
        from audit_supermimic_model import projection_adapter
        adapter = self.adapter()
        snapshot = {name: getattr(adapter, name) for name in
            ("layout", "history_terms", "critic_terms", "qpos_quaternion_slices", "joint_projection_metadata")}
        collection = {"step_dt": .02, "cycle_time": 6., "phase_ratios": [0., .4, .5, .95, 1.],
                      "board_quaternion_slice": None, "command_config": {}}
        off = projection_adapter(snapshot, collection)
        self.assertFalse(off.joint_consistency)
        on = projection_adapter(snapshot, collection, snapshot["joint_projection_metadata"])
        self.assertTrue(on.joint_consistency)
        self.assertEqual(on.joint_projection_metadata["spec_sha256"], adapter.joint_projection_metadata["spec_sha256"])
        bad = copy.deepcopy(snapshot["joint_projection_metadata"]); bad["spec_sha256"] = "bad"
        with self.assertRaises(ValueError): projection_adapter(snapshot, collection, bad)


if __name__ == "__main__":
    unittest.main()
