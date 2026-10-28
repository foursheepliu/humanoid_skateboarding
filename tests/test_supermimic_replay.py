"""CPU tests: no MuJoCo, CUDA initialization or real optimizer updates."""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch

# Load only these independent modules, so a test never imports a simulator or
# accidentally initializes GPU task plugins from the package entry points.
_path = Path(__file__).resolve().parents[1] / "src/mjlab_husky/supermimic"
_package = ModuleType("_coupled_replay_test")
_package.__path__ = [str(_path)]
sys.modules[_package.__name__] = _package
_replay = importlib.import_module(_package.__name__ + ".replay")
_adapter = importlib.import_module(_package.__name__ + ".env_adapter")
EpisodeSequenceReplay, TransitionBatch = _replay.EpisodeSequenceReplay, _replay.TransitionBatch
CoupledStateAdapter = _adapter.CoupledStateAdapter


class FakeEnvironment:
    def __init__(self, num_envs=2):
        n = self.num_envs = num_envs
        self.device, self.step_dt, self.cycle_time = "cpu", 0.02, 6.0
        self.cfg = SimpleNamespace(phase_ratios=[0.0, 0.4, 0.5, 0.95, 1.0])
        self.sim = SimpleNamespace(
            data=SimpleNamespace(qpos=torch.zeros(n, 14), qvel=torch.zeros(n, 12),
                                 act=torch.zeros(n, 0), ctrl=torch.zeros(n, 3)),
            model=SimpleNamespace(body_mass=torch.ones(1, 2)),
            mj_model=SimpleNamespace(jnt_type=[0, 0], jnt_qposadr=[0, 7]))
        self.sim.data.qpos[:, 3] = self.sim.data.qpos[:, 10] = 1
        self.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(n, 23)))
        self.action_manager = SimpleNamespace(total_action_dim=3, action=torch.zeros(n, 3),
                                              prev_action=torch.zeros(n, 3), prev_prev_action=torch.zeros(n, 3))
        self.term = SimpleNamespace(command_b=torch.ones(n, 2), heading_ref_b=torch.zeros(n),
                                    is_heading_env=torch.ones(n, dtype=torch.bool), is_standing_env=torch.zeros(n, dtype=torch.bool))
        self.command_manager = SimpleNamespace(get_term=lambda name: self.term)
        self.contact_phase = torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1)
        self.phase_length_buf = torch.zeros(n)
        self.episode_length_buf = torch.zeros(n)
        self.still = torch.zeros(n, dtype=torch.bool)
        self.body_bezier_buffers = {"push2steer_start_pos_b": torch.zeros(n, 2, 3)}
        self.obs_buf = {"policy": torch.arange(40.).repeat(n, 1), "critic": torch.zeros(n, 10)}
        self.observation_calls = 0
        self.observation_manager = SimpleNamespace(
            _group_obs_term_names={"policy": ["command", "body", "actions", "phase"],
                                  "critic": ["command", "actions", "phase", "contact_phase"]},
            _group_obs_term_dim={"policy": [(10,), (10,), (15,), (5,)],
                                "critic": [(2,), (3,), (1,), (4,)]},
            _group_obs_term_cfgs={"policy": [SimpleNamespace(history_length=5, scale=(2., 1.))]
                                          + [SimpleNamespace(history_length=5, scale=None)] * 3,
                                  "critic": [SimpleNamespace(history_length=0, scale=(2., 1.))]
                                          + [SimpleNamespace(history_length=0, scale=None)] * 3},
            compute=self._compute_obs)
        self.steps = 0

    def _compute_obs(self, update_history=False):
        self.observation_calls += 1
        return self.obs_buf

    def _reset_idx(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.sim.data.qpos[ids, 0] = -1000
        self.robot.data.joint_pos[ids] = -10
        self.phase_length_buf[ids] = self.episode_length_buf[ids] = 0

    def step(self, actions):
        self.steps += 1
        self.sim.data.qpos[:, 0] += 1
        self.robot.data.joint_pos += 0.1
        self.episode_length_buf += 1
        self.phase_length_buf += 1
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.steps % 3 == 0:
            done[0] = True
            self._reset_idx(torch.tensor([0]))
        self.obs_buf = {"policy": self.obs_buf["policy"] + 1, "critic": self.obs_buf["critic"] + 1}
        return self.obs_buf, torch.arange(self.num_envs).float(), done, torch.zeros_like(done), {}


class AdapterTests(unittest.TestCase):
    def test_terminal_physics_preserved_without_observation_rng_or_history_update(self):
        env = FakeEnvironment()
        adapter = CoupledStateAdapter(env)
        for _ in range(3):
            transition = adapter.capture_step(torch.zeros(2, 3))
        q = adapter.layout["qpos"]
        self.assertEqual(float(transition.next_state[0, q.start]), 3.0)
        self.assertEqual(float(env.sim.data.qpos[0, 0]), -1000.0)
        self.assertFalse(bool(transition.valid_next[0]))
        self.assertTrue(bool(transition.terminated[0]))
        self.assertTrue(bool(transition.valid_next[1]))
        self.assertEqual(env.observation_calls, 0)
        self.assertEqual(int(adapter._amp_history[0].count_nonzero()), 0)
        adapter.close()
        self.assertFalse(hasattr(env, "_supermimic_adapter"))

    def test_projection_term_major_history_and_gradient(self):
        adapter = CoupledStateAdapter(FakeEnvironment())
        previous = adapter.observe()
        predicted = (previous + 100).detach().requires_grad_(True)
        result = adapter.project(previous, predicted, torch.ones(2, 3))
        for term in adapter.history_terms:
            old = previous[:, term["slice"]].reshape(2, 5, term["frame_dim"])
            new = result[:, term["slice"]].reshape(2, 5, term["frame_dim"])
            torch.testing.assert_close(new[:, :-1], old[:, 1:])
            targets = {"command": torch.tensor([[2., 1.]]).repeat(2, 1),
                       "actions": torch.ones(2, 3), "phase": torch.full((2, 1), 0.02 / 6)}
            target = targets.get(term["name"], old[:, -1] + 100)
            torch.testing.assert_close(new[:, -1], target)
        for term in adapter.critic_terms:
            target = targets.get(term["name"], torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1))
            torch.testing.assert_close(result[:, term["slice"]], target)
        torch.testing.assert_close(result[:, adapter.layout["action"]], torch.ones(2, 3))
        for sl in adapter.qpos_quaternion_slices:
            torch.testing.assert_close(result[:, sl].norm(dim=-1), torch.ones(2))
        result[:, adapter.layout["amp_joints"]].sum().backward()
        self.assertIsNotNone(predicted.grad)
        self.assertTrue(bool(torch.isfinite(predicted.grad).all()))
        adapter.close()

    def test_eight_step_projected_bptt_reaches_actor_without_inplace_errors(self):
        adapter = CoupledStateAdapter(FakeEnvironment())
        actor = torch.nn.Linear(adapter.policy_dim, adapter.action_dim)
        world = torch.nn.Linear(adapter.state_dim + adapter.action_dim, adapter.state_dim)
        state = adapter.observe().detach()
        loss = torch.zeros(())
        for _ in range(8):
            action = actor(adapter.state_to_observations(state)["policy"])
            predicted = world(torch.cat((state, action), -1))
            state = adapter.project(state, predicted, action)
            loss = loss + state.square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in actor.parameters()))
        adapter.close()


class ReplayTests(unittest.TestCase):
    def batch(self, time, context, dones=None):
        n = len(context)
        state = torch.stack((torch.full((n,), float(time)), torch.arange(n).float()), -1)
        done = torch.zeros(n, dtype=torch.bool) if dones is None else torch.tensor(dones, dtype=torch.bool)
        next_state = state + torch.tensor([1., 0.])
        next_state[done, 0] = 12345  # Distinct pre-reset diagnostic target.
        return TransitionBatch(state, torch.full((n, 1), float(time)), next_state,
                               torch.full((n,), float(time)), done, ~done, context, done)

    def test_ring_wrap_sequences_do_not_cross_resets(self):
        replay = EpisodeSequenceReplay(2, 2, 1, 1, 7)
        context = torch.tensor([[1.], [2.]])
        for t in range(15):
            replay.add(self.batch(t, context, [t % 4 == 3, t % 5 == 4]))
        batch = replay.sample(500, 3)
        self.assertEqual(batch.states.shape, (500, 4, 2))
        self.assertFalse(bool(batch.dones[:, :-1].any()))
        self.assertTrue(bool((batch.states[:, 0, 0] >= 8).all()))
        torch.testing.assert_close(batch.states[:, 1:3, 0] - batch.states[:, :2, 0], torch.ones(500, 2))
        if bool(batch.dones[:, -1].any()):
            ended = batch.dones[:, -1]
            self.assertTrue(bool((batch.states[ended, -1, 0] == 12345).all()))
            self.assertFalse(bool(batch.valid[ended, -1].any()))
        torch.testing.assert_close(batch.context[:, :, 0], batch.states[:, :-1, 1] + 1)
        self.assertEqual(replay._context_bank.shape[0], 1)

    def test_checkpoint_resume_breaks_trajectory_and_keeps_context_versions(self):
        replay = EpisodeSequenceReplay(2, 2, 1, 1, 8)
        old = torch.ones(2, 1)
        for t in range(4):
            replay.add(self.batch(t, old))
        restored = EpisodeSequenceReplay(2, 2, 1, 1, 8)
        restored.load_state_dict(replay.state_dict())
        restored.break_continuity()
        new = torch.full((2, 1), 5.)
        for t in range(10, 14):
            restored.add(self.batch(t, new))
        batch = restored.sample(500, 3)
        self.assertFalse(bool(((batch.states[:, :-1, 0] < 5).any(1) & (batch.states[:, :-1, 0] > 5).any(1)).any()))
        expected = torch.where(batch.states[:, 0, 0] < 5, 1., 5.)[:, None].expand(-1, 3)
        torch.testing.assert_close(batch.context[..., 0], expected)
        self.assertEqual(restored._context_bank.shape[0], 2)

    def test_reject_insufficient_or_all_reset_windows(self):
        replay = EpisodeSequenceReplay(1, 2, 1, 0, 4)
        context = torch.empty(1, 0)
        with self.assertRaises(ValueError):
            replay.sample(2, 3)
        for t in range(4):
            replay.add(self.batch(t, context, [True]))
        with self.assertRaises(ValueError):
            replay.sample(2, 2)

    def test_environment_holdout_filter(self):
        replay = EpisodeSequenceReplay(4, 2, 1, 0, 8)
        context = torch.empty(4, 0)
        for t in range(5):
            replay.add(self.batch(t, context))
        batch = replay.sample(200, 3, env_ids=[2, 3])
        self.assertTrue(bool((batch.states[..., 1] >= 2).all()))
        with self.assertRaises(ValueError):
            replay.sample(1, 2, env_ids=[])

    def test_filtered_candidate_cache_tracks_edits_adds_and_resume(self):
        replay = EpisodeSequenceReplay(4, 2, 1, 0, 8)
        context = torch.empty(4, 0)
        for t in range(5):
            replay.add(self.batch(t, context))
        ids = torch.tensor([0, 1])
        first = replay._filtered_candidates(3, ids)
        self.assertIs(first, replay._filtered_candidates(3, ids))
        ids[:] = torch.tensor([2, 3])
        edited = replay._filtered_candidates(3, ids)
        self.assertTrue(bool((edited[:, 1] >= 2).all()))
        self.assertEqual(replay.host_allocated_bytes, replay.allocated_bytes)
        self.assertEqual(replay.device_allocated_bytes, 0)
        replay.add(self.batch(5, context))
        self.assertFalse(replay._filtered_candidate_cache)
        replay._filtered_candidates(3, ids)
        replay.break_continuity()
        self.assertFalse(replay._filtered_candidate_cache)
        snapshot = replay.state_dict()
        self.assertTrue(all(t.device.type == "cpu" for t in snapshot["tensors"].values()))
        # Snapshot is a copy, not a reference to subsequent mutable storage.
        before = snapshot["tensors"]["states"].clone()
        replay.states.add_(1)
        torch.testing.assert_close(before, snapshot["tensors"]["states"])
        replay.load_state_dict(snapshot)
        self.assertFalse(replay._candidate_cache)


if __name__ == "__main__":
    unittest.main()
