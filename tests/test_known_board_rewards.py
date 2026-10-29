"""Board-S reward/source/collection contracts; Torch + fake env, no simulation."""

from __future__ import annotations

import ast
from copy import deepcopy
from itertools import product
import math
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from mjlab_husky.supermimic import known_board_rewards as board_rewards
from mjlab_husky.supermimic.adapter import SkateWorldModelAdapter, StateLayout
from mjlab_husky.supermimic.known_board_rewards import (
    BoardRewardObserver, KnownBoardRewards, NextPhysicalRewards, RewardPhysics,
    capture_real_reward_physics, inverse_rotate,
)
from test_supermimic_adapter import _full_task


REPO = Path(__file__).resolve().parents[1]


def _physical(**overrides):
    values = dict(push_weight=3., flat_weight=5., push_std=.5, flat_std=.3,
                  step_dt=.02, push_scale_by_dt=True, flat_scale_by_dt=True,
                  cycle_time=6., phase_ratios=(0., .4, .5, .95, 1.))
    values.update(overrides)
    return NextPhysicalRewards(**values)


def _layout():
    return StateLayout({"phase_length_buf": (), "command.skate.command_b": (2,),
                        "contact_phase": (4,), "skateboard.root_link_quat_w": (4,),
                        "skateboard.root_link_lin_vel_w": (3,), "skateboard.joint_pos": (7,)})


def _original_functions():
    """Execute exactly two original function ASTs, avoiding optional mjlab imports."""
    module = ModuleType("mjlab_husky.tasks.skater.mdp.rewards")
    module.__dict__.update(torch=torch, _DEFAULT_ASSET_CFG=NS(name="robot"))
    source = REPO / "src/mjlab_husky/tasks/skater/mdp/rewards.py"
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"push_skateboard_lin_vel", "board_flat"}]
    assert len(functions) == 2
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ast.fix_missing_locations(future)
    exec(compile(ast.Module(body=[future, *functions], type_ignores=[]), str(source), "exec"), module.__dict__)
    return module


def _matrix_inverse_rotate(q, vector):
    # Independent full rotation matrix oracle for unit WXYZ quaternions.
    w, x, y, z = q.unbind(-1)
    rotation = torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                            2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                            2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1)
    return torch.einsum("...ji,...j->...i", rotation.reshape(*q.shape[:-1], 3, 3), vector)


def _live_configuration():
    original = _original_functions()
    push = NS(func=original.push_skateboard_lin_vel, weight=4.,
              params={"asset_cfg": NS(name="skateboard"), "command_name": "skate", "std": .7})
    flat = NS(func=original.board_flat, weight=5., params={"std": .3})
    raw = NS(step_dt=.03, cycle_time=6., cfg=NS(phase_ratios=[0., .4, .5, .95, 1.]),
             phase_ratios=torch.tensor([[0., .4, .5, .95, 1.]]),
             push_reward_manager=NS(get_term_cfg={"push_skateboard_lin_vel": push}.__getitem__, _scale_by_dt=False),
             reg_reward_manager=NS(get_term_cfg={"board_flat": flat}.__getitem__, _scale_by_dt=True))
    return original, raw, push, flat


def _original_phase_class():
    """Keep the actual clock and contact-phase source, omit later FK caches."""
    source = REPO / "src/mjlab_husky/envs/g1_skate_rl_env.py"
    tree = ast.parse(source.read_text())
    functions = {node.name: deepcopy(node) for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name in {"_get_phase", "_resample_contact_phases"}}
    resample = functions["_resample_contact_phases"]
    cut = next(i for i, node in enumerate(resample.body) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "body_pos_w" for target in node.targets))
    resample.body = resample.body[:cut]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=list(functions.values()), type_ignores=[]), str(source), "exec"), namespace)
    return type("SourcePhase", (), {name: namespace[name] for name in functions})


class ComponentTests(unittest.TestCase):
    def test_original_source_full_rotation_xyz_weights_dt_and_phase(self):
        original = _original_functions()
        root_half = 2**-.5
        q = torch.tensor([[1., 0., 0., 0.], [root_half, 0., root_half, 0.],
                          [.7, .2, -.3, .5], [1., 0., 0., 0.], [1., 0., 0., 0.]], dtype=torch.float64)
        q = q / q.norm(dim=-1, keepdim=True)
        velocity = torch.tensor([[.8, .1, .2], [0., 0., -1.], [.2, -.4, .3],
                                 [1., 0., 0.], [0., 0., 0.]], dtype=torch.float64)
        command = torch.tensor([[1., .9], [1., -.9], [.7, .1], [1., 0.], [0., 0.]], dtype=torch.float64)
        gamma = torch.tensor([.01, -.1, .02, .3, -.03], dtype=torch.float64)
        phase = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 0.]], dtype=torch.float64)
        body_velocity = _matrix_inverse_rotate(q, velocity)
        torch.testing.assert_close(inverse_rotate(q, velocity), body_velocity, rtol=0, atol=3e-16)
        asset = NS(data=NS(root_link_lin_vel_b=body_velocity, joint_pos=gamma[:, None]))
        env = NS(scene={"skateboard": asset}, skateboard=asset,
                 command_manager=NS(get_command=lambda _name: command),
                 contact_phase=torch.cat((phase, torch.zeros(5, 2, dtype=torch.float64)), -1))
        for push_dt, flat_dt in product((False, True), repeat=2):
            known = _physical(push_weight=4., flat_weight=7., step_dt=.031,
                              push_scale_by_dt=push_dt, flat_scale_by_dt=flat_dt)
            source_push = original.push_skateboard_lin_vel(env, known.push_std, "skate", NS(name="skateboard"))
            source_flat = original.board_flat(env, known.flat_std)
            actual_push, actual_flat = known.components(command, RewardPhysics(q, velocity, gamma, phase))
            torch.testing.assert_close(actual_push, source_push*4.*(.031 if push_dt else 1.)*phase[:, 0])
            torch.testing.assert_close(actual_flat, source_flat*7.*(.031 if flat_dt else 1.))
            self.assertAlmostEqual(float(actual_push[1]), 4.*(.031 if push_dt else 1.))
            self.assertEqual(float(actual_flat[3]), 0.)  # steer alone disables flat
            self.assertGreater(float(actual_flat[4]), 0.)  # still/transition retains flat
        # Y/Z errors are both used, and the second command coordinate is not.
        identity = q[:1].repeat(3, 1)
        v = torch.tensor([[1., 0., 0.], [1., .2, 0.], [1., 0., .3]], dtype=torch.float64)
        c = command[:1].repeat(3, 1)
        result, _ = _physical(push_weight=1., push_scale_by_dt=False).components(
            c, RewardPhysics(identity, v, torch.zeros(3), torch.tensor([[1., 0.]]).repeat(3, 1)))
        torch.testing.assert_close(result, torch.exp(-torch.tensor([0., .04, .09], dtype=torch.float64)/.25))

    def test_configuration_and_schema_are_live_validated_without_computing_rewards(self):
        original, env, _, _ = _live_configuration()
        with patch.dict(sys.modules, {original.__name__: original}):
            known = KnownBoardRewards.from_env(NS(unwrapped=env), _layout())
        metadata = known.metadata()
        self.assertEqual(metadata["weights"], [4., 5.])
        self.assertEqual(metadata["std"], [.7, .3])
        self.assertEqual(metadata["manager_scale_by_dt"], [False, True])
        self.assertEqual(metadata["step_dt"], .03)
        self.assertEqual(metadata["snapshot_values_per_env"], 10)
        self.assertFalse(metadata["continuation_multiplier"])
        self.assertFalse(metadata["action_cost_decomposition"])
        self.assertEqual(metadata["state_fields"]["command.skate.command_b"], [2])
        self.assertEqual(metadata["state_dimension"], known.layout.dim)
        self.assertEqual(known.physical.phase_ratios, (0., .4, .5, .95, 1.))
        self.assertEqual(known.physical.push_scale, 1.)
        self.assertEqual(known.physical.flat_scale, .03)
        owner = torch.nn.Module()
        owner.known = known
        self.assertEqual(owner.state_dict(), {})

    def test_rejects_source_parameter_phase_and_scaling_drift(self):
        for mutation in ("identity", "module", "asset", "command", "params", "std", "weight", "dt", "flag", "phase"):
            with self.subTest(mutation=mutation):
                original, raw, push, flat = _live_configuration()
                if mutation == "identity":
                    def impostor(*args):
                        raise AssertionError("factory executed a reward")
                    impostor.__name__, impostor.__module__ = push.func.__name__, push.func.__module__
                    push.func = impostor
                elif mutation == "module":
                    original.push_skateboard_lin_vel.__module__ = "another_module"
                elif mutation == "asset":
                    push.params["asset_cfg"].name = "robot"
                elif mutation == "command":
                    push.params["command_name"] = "other"
                elif mutation == "params":
                    flat.params["extra"] = 1
                elif mutation == "std":
                    push.params["std"] = 0
                elif mutation == "weight":
                    flat.weight = math.nan
                elif mutation == "dt":
                    raw.step_dt = 0
                elif mutation == "flag":
                    raw.reg_reward_manager._scale_by_dt = 1
                else:
                    raw.phase_ratios[0, 1] = .3
                with patch.dict(sys.modules, {original.__name__: original}), self.assertRaises(ValueError):
                    KnownBoardRewards.from_env(raw, _layout())
        for key, value in (("command.skate.command_b", (3,)), ("phase_length_buf", (1,)),
                           ("skateboard.root_link_quat_w", (3,)), ("skateboard.joint_pos", (0,))):
            shapes = dict(_layout().shapes)
            shapes[key] = value
            with self.assertRaises(ValueError):
                KnownBoardRewards(StateLayout(shapes), _physical())

    def test_manager_sanitization_precedes_push_gate_and_flat_is_nonsteer(self):
        physical = RewardPhysics(torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1),
            torch.tensor([[math.nan, 0., 0.], [math.inf, 0., 0.], [0., 0., 0.]]),
            torch.tensor([math.nan, math.inf, -.02]), torch.tensor([[1., 0.], [0., 1.], [0., 0.]]))
        push, flat = _physical().components(torch.tensor([[1., 0.]]).repeat(3, 1), physical)
        self.assertTrue(torch.isfinite(push).all())
        self.assertTrue(torch.isfinite(flat).all())
        torch.testing.assert_close(push, torch.zeros(3))
        self.assertEqual(float(flat[0]), 0.)
        self.assertEqual(float(flat[1]), 0.)
        self.assertGreater(float(flat[2]), 0.)


class ProjectionTests(unittest.TestCase):
    def test_reward_phase_matches_original_clock_and_contact_source_at_boundaries_and_standing(self):
        cls = _original_phase_class()
        clocks = torch.tensor([0, 1, 118, 119, 120, 148, 149, 150, 283, 284, 285, 298, 299, 300, 449])
        for speed in (0., .099, .1, 1.):
            raw = cls()
            raw.phase_length_buf = clocks.clone() + 1  # Actual step increments before _compute_contact.
            raw.step_dt, raw.cycle_time = .02, 6.
            command = torch.tensor([[speed, -.2]]).repeat(len(clocks), 1)
            raw.still = command[:, 0] < .1
            raw.contact_phase = torch.full((len(clocks), 4), 77.)
            raw.phase_ratios = torch.tensor([[0., .4, .5, .95, 1.]]).repeat(len(clocks), 1)
            raw._resample_contact_phases()
            expected = raw.contact_phase[:, :2]
            actual = _physical().reward_phase(clocks.float(), command)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if speed < .1:
                self.assertFalse(actual.any())
                self.assertEqual(int(raw.phase_length_buf[0]), 0)
                self.assertEqual(int(raw.phase_length_buf[7]), 0)

    def test_minimal_and_full_adapter_match_values_and_packed_VJP_at_phase_boundaries(self):
        env, observations, amp = _full_task()
        adapter = SkateWorldModelAdapter(env, observations, amp)
        initial = adapter.capture(observations, amp)
        known = KnownBoardRewards(adapter.layout, _physical())
        cases = (([0., 118.], [1., 1.]), ([119., 120.], [1., 1.]),
                 ([148., 149.], [1., 1.]), ([150., 283.], [1., 1.]),
                 ([284., 285.], [1., 1.]), ([298., 299.], [1., 1.]),
                 ([300., 449.], [1., 1.]), ([0., 150.], [0., .099]),
                 ([149., 150.], [.1, .1]))
        for clocks, speed in cases:
            for quaternion in (torch.tensor([[2., .2, .3, .1], [0., 0., 0., 0.]]),
                               torch.tensor([[-2., -.2, -.3, -.1], [1e-9, 0., 0., 0.]])):
                with self.subTest(clock=clocks, speed=speed, first_sign=float(quaternion[0, 0])):
                    fields = adapter.layout.unpack(initial)
                    fields["phase_length_buf"] = torch.tensor(clocks)
                    fields["command.skate.command_b"] = torch.tensor([[speed[0], .2], [speed[1], -.3]])
                    # Force a post-reward resample: full projected commands differ
                    # from the OLD command used by both reward paths.
                    fields["command.skate.time_left"] = torch.zeros(2)
                    state = adapter.layout.pack(fields).detach().requires_grad_()
                    raw = adapter.layout.unpack(state.detach())
                    raw["skateboard.root_link_quat_w"] = quaternion
                    raw["skateboard.root_link_lin_vel_w"] = torch.tensor([[.6, .1, .2], [.4, -.1, .3]])
                    joints = raw["skateboard.joint_pos"].clone()
                    joints[:, 0] = torch.tensor([.03, -.02])
                    raw["skateboard.joint_pos"] = joints
                    raw["contact_phase"] = torch.full((2, 4), 77.)
                    raw["phase_length_buf"] = torch.full((2,), 9999.)
                    raw["command.skate.command_b"] = torch.full((2, 2), -999.)
                    raw_next = adapter.layout.pack(raw).detach().requires_grad_()
                    action = torch.zeros(2, 23, requires_grad=True)
                    projected = adapter.project_next(state, raw_next, action, exogenous={
                        "command_uniforms": torch.zeros(2, adapter.command_noise_dim),
                        "sensor_uniforms": torch.zeros(2, adapter.sensor_noise_dim)})
                    p = adapter.layout.unpack(projected)
                    old_command = adapter.layout.get(state, "command.skate.command_b")
                    full = known.physical(old_command, RewardPhysics(
                        p["skateboard.root_link_quat_w"], p["skateboard.root_link_lin_vel_w"],
                        p["skateboard.joint_pos"][:, 0], p["contact_phase"][:, :2]))
                    minimal = known(state, raw_next)
                    torch.testing.assert_close(minimal, full, rtol=0, atol=0)
                    cotangent = torch.tensor([.7, -1.3])
                    full_grads = torch.autograd.grad((full*cotangent).sum(), (state, raw_next, action),
                                                     retain_graph=True, allow_unused=True)
                    minimal_grads = torch.autograd.grad((minimal*cotangent).sum(), (state, raw_next, action),
                                                        allow_unused=True)
                    for a, b in zip(full_grads, minimal_grads):
                        if a is None or b is None:
                            self.assertTrue((a is None or not a.any()) and (b is None or not b.any()))
                        else:
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                            self.assertTrue(torch.isfinite(a).all())
                    for name in ("contact_phase", "phase_length_buf", "command.skate.command_b"):
                        torch.testing.assert_close(adapter.layout.get(minimal_grads[1], name),
                            torch.zeros_like(adapter.layout.get(raw_next, name)), rtol=0, atol=0)
        self.assertEqual(env.observation_manager.compute_calls, 0)

    def test_task_reward_time_stage_source_order_is_before_reset_and_command_events(self):
        source = REPO / "src/mjlab_husky/envs/g1_skate_rl_env.py"
        tree = ast.parse(source.read_text())
        step = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "step")
        calls = {ast.unparse(node.func): node.lineno for node in ast.walk(step) if isinstance(node, ast.Call)}
        self.assertLess(calls["self._compute_contact"], calls["self.reg_reward_manager.compute"])
        self.assertLess(calls["self.reg_reward_manager.compute"], calls["self._reset_idx"])
        self.assertLess(calls["self._reset_idx"], calls["self.command_manager.compute"])
        self.assertLess(calls["self.command_manager.compute"], calls["self.event_manager.apply"])
        self.assertLess(calls["self.event_manager.apply"], calls["self.observation_manager.compute"])


def _observer_fixture(mode="class"):
    class Manager:
        def __init__(self):
            self.calls, self.dts, self.events = 0, [], []
            self.result = torch.tensor([.3, -.2, .7])
        def compute(self, dt):
            self.calls += 1
            self.dts.append(dt)
            self.events.append("original_compute")
            return self.result
    manager = Manager()
    if mode == "instance_plain":
        def custom(dt):
            return Manager.compute(manager, dt)
        manager.compute = custom
    elif mode == "instance_bound":
        def custom(self, dt):
            return Manager.compute(self, dt)
        manager.compute = MethodType(custom, manager)
    command = torch.tensor([[1., .2], [1., -.3], [0., 0.]], requires_grad=True)
    raw = NS(reg_reward_manager=manager,
        command_manager=NS(get_command=lambda _name: command),
        skateboard=NS(data=NS(root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1),
            root_link_lin_vel_w=torch.tensor([[.2, 0., 0.], [.3, .1, 0.], [.4, 0., .2]]),
            joint_pos=torch.tensor([[.01, 9.], [.02, 9.], [-.03, 9.]]))),
        contact_phase=torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., 1., 0., 0.]]))
    return raw, command, KnownBoardRewards(_layout(), _physical())


class ObserverTests(unittest.TestCase):
    def test_same_return_once_snapshot_before_reset_resample_and_full_batch_overwrite(self):
        for mode in ("class", "instance_plain", "instance_bound"):
            with self.subTest(mode=mode):
                raw, command, known = _observer_fixture(mode)
                manager, original = raw.reg_reward_manager, raw.reg_reward_manager.compute
                previous = manager.__dict__.get("compute")
                expected_snapshot = capture_real_reward_physics(raw)
                expected = known.physical(command.detach().clone(), expected_snapshot)
                real_read = board_rewards.capture_real_reward_physics
                def read(r):
                    manager.events.append("snapshot")
                    return real_read(r)
                with patch.object(board_rewards, "capture_real_reward_physics", read):
                    with known.observe_step(NS(unwrapped=raw)) as observer:
                        result = manager.compute(.02)
                        self.assertIs(result, manager.result)
                        self.assertEqual((manager.calls, observer.calls, observer.original_calls), (1, 1, 1))
                        with self.assertRaises(RuntimeError):
                            observer.finish()  # Must restore the hook before accepting labels.
                        manager.events.append("reset")
                        raw.skateboard.data.root_link_quat_w.zero_()
                        raw.skateboard.data.root_link_lin_vel_w.fill_(999.)
                        raw.skateboard.data.joint_pos.fill_(999.)
                        raw.contact_phase.zero_()
                        manager.events.append("resample")
                        with torch.no_grad():
                            command.fill_(99.)
                        manager.events.append("post_reward_event")
                captured = observer.finish()
                self.assertEqual(manager.events, ["original_compute", "snapshot", "reset", "resample", "post_reward_event"])
                self.assertIs(captured, observer.value)
                torch.testing.assert_close(captured, expected, rtol=0, atol=0)
                torch.testing.assert_close(observer.old_command, torch.tensor([[1., .2], [1., -.3], [0., 0.]]))
                self.assertFalse(observer.old_command.requires_grad)
                for name in ("quaternion_wxyz", "linear_velocity_w", "gamma", "push_steer_phase"):
                    torch.testing.assert_close(getattr(observer.snapshot, name), getattr(expected_snapshot, name), rtol=0, atol=0)
                    self.assertFalse(getattr(observer.snapshot, name).requires_grad)
                self.assertEqual(sum(x.numel() for x in (observer.snapshot.quaternion_wxyz,
                    observer.snapshot.linear_velocity_w, observer.snapshot.gamma, observer.snapshot.push_steer_phase)), 30)
                if mode == "class":
                    self.assertNotIn("compute", manager.__dict__)
                    self.assertEqual(manager.compute, original)
                else:
                    self.assertIs(manager.compute, previous)

    def test_old_command_is_owned_even_if_original_tensor_changes_before_compute(self):
        raw, command, known = _observer_fixture()
        expected_command = command.detach().clone()
        with known.observe_step(raw) as observer:
            with torch.no_grad():
                command.fill_(99.)
            raw.reg_reward_manager.compute(.02)
        torch.testing.assert_close(observer.old_command, expected_command, rtol=0, atol=0)
        torch.testing.assert_close(observer.finish(), known.physical(expected_command, observer.snapshot), rtol=0, atol=0)

    def test_missing_extra_bad_dt_and_capture_error_reject_without_skipping_original(self):
        for failure in ("missing", "extra", "bad_dt", "capture"):
            with self.subTest(failure=failure):
                raw, _, known = _observer_fixture()
                original = raw.reg_reward_manager.compute
                observer = BoardRewardObserver(known, raw)
                read_patch = patch.object(board_rewards, "capture_real_reward_physics",
                    side_effect=RuntimeError("capture failed")) if failure == "capture" else patch.object(
                    board_rewards, "capture_real_reward_physics", wraps=capture_real_reward_physics)
                with read_patch as reader:
                    with observer:
                        for _ in range(0 if failure == "missing" else 2 if failure == "extra" else 1):
                            self.assertIs(raw.reg_reward_manager.compute(.01 if failure == "bad_dt" else .02),
                                          raw.reg_reward_manager.result)
                        # Capture errors do not abort ordinary resets/events.
                        raw.reg_reward_manager.events.append("ordinary_step_continued")
                self.assertEqual(raw.reg_reward_manager.compute, original)
                self.assertNotIn("compute", raw.reg_reward_manager.__dict__)
                calls = 0 if failure == "missing" else 2 if failure == "extra" else 1
                self.assertEqual((raw.reg_reward_manager.calls, observer.calls, observer.original_calls), (calls, calls, calls))
                self.assertEqual(reader.call_count, 0 if failure in ("missing", "bad_dt") else 1)
                with self.assertRaises(RuntimeError):
                    observer.finish()

    def test_exceptional_context_and_original_compute_exception_restore_both_method_kinds(self):
        for mode, original_raises in product(("class", "instance_plain", "instance_bound"), (False, True)):
            with self.subTest(mode=mode, original_raises=original_raises):
                raw, _, known = _observer_fixture(mode)
                manager = raw.reg_reward_manager
                if original_raises:
                    def fail(dt):
                        manager.calls += 1
                        raise LookupError("original failed")
                    manager.compute = fail
                original, had_instance = manager.compute, "compute" in manager.__dict__
                previous = manager.__dict__.get("compute")
                observer = known.observe_step(raw)
                with self.assertRaises(LookupError if original_raises else ArithmeticError):
                    with observer:
                        manager.compute(.02)
                        raise ArithmeticError("body failed")
                self.assertTrue(observer.restored)
                self.assertEqual("compute" in manager.__dict__, had_instance)
                if had_instance:
                    self.assertIs(manager.compute, previous)
                else:
                    self.assertEqual(manager.compute, original)
                self.assertEqual(manager.calls, 1)
                if original_raises:
                    with self.assertRaises(RuntimeError):
                        observer.finish()

    def test_finish_detects_later_hook_replacement_and_invalid_entry_leaves_method_untouched(self):
        raw, _, known = _observer_fixture()
        with known.observe_step(raw) as observer:
            raw.reg_reward_manager.compute(.02)
        raw.reg_reward_manager.compute = lambda dt: None
        with self.assertRaises(RuntimeError):
            observer.finish()
        for command in (None, torch.zeros(3, 3)):
            raw, _, known = _observer_fixture()
            raw.command_manager.get_command = lambda _name: command
            original = raw.reg_reward_manager.compute
            with self.assertRaises(ValueError):
                with known.observe_step(raw):
                    self.fail("invalid entry succeeded")
            self.assertEqual(raw.reg_reward_manager.compute, original)
            self.assertNotIn("compute", raw.reg_reward_manager.__dict__)


if __name__ == "__main__":
    unittest.main()
