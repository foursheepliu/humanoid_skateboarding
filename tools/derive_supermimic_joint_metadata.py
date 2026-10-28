"""Derive and independently verify constant joint metadata from real CPU replay.

The training environments establish the unique coordinate correspondence.
Held-out environments only verify it. Default poses come from the frozen asset
configuration, not fitted future physical values. No input file is modified.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET


def source_joint_defaults(root):
    import torch
    asset = root / "src/mjlab_husky/asset_zoo/robots/skateboard"
    source = asset / "g1_skater_constants.py"
    tree = ast.parse(source.read_text())
    initial = next(node.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PUSH_INIT_KEYFRAME" for target in node.targets))
    dictionaries = {key: ast.literal_eval(next(item.value for item in initial.keywords if item.arg == key))
                    for key in ("joint_pos", "joint_vel")}
    xml = asset / "xmls/g1.xml"
    names = [joint.attrib["name"] for body in ET.parse(xml).getroot().findall("worldbody")
             for joint in body.iter("joint") if "name" in joint.attrib]
    if len(names) != 23 or len(set(names)) != 23:
        raise ValueError("Require 23 unique natural-order G1 XML joints")
    defaults = {}
    for key, configured in dictionaries.items():
        values = []
        for name in names:
            matches = [value for expression, value in configured.items() if re.fullmatch(expression, name)]
            if len(matches) > 1:
                raise ValueError("Ambiguous source default joint expressions")
            values.append(matches[0] if matches else 0.0)
        defaults[key] = torch.tensor(values, dtype=torch.float32)
    return names, defaults, (source, xml)


def replay_rows(corpus, env_ids):
    import torch
    layout, tensors = corpus["layout"], corpus["replay_state_dict"]["tensors"]
    length = min(corpus["replay_state_dict"]["total_steps"], corpus["replay_state_dict"]["shape"][-1])
    valid = tensors["valid"][:length, env_ids].bool()
    terms = {term["name"]: term for term in corpus["critic_terms"]}
    groups = corpus["physics_groups"]
    slices = {"amp": layout["amp_joints"], "qpos": layout["qpos"], "qvel": layout["qvel"],
              "critic_pos": terms["joint_pos"]["slice"], "critic_vel": terms["joint_vel"]["slice"]}
    result = {}
    for name, sl in slices.items():
        values = []
        for field in ("states", "next_states"):
            item = tensors[field][:length, :, sl][:, env_ids]
            item = item[valid] if field == "next_states" else item.flatten(0, 1)
            if name in ("qpos", "qvel"):
                indices = groups["robot_joint_angle" if name == "qpos" else "robot_joint_velocity"]["indices"]
                item = item[:, indices]
            values.append(item)
        result[name] = torch.cat(values)
        if not bool(torch.isfinite(result[name]).all()):
            raise FloatingPointError("Joint metadata evidence contains nonfinite real values")
    result["stored_state_rows"] = length*len(env_ids)
    result["valid_next_rows"] = int(valid.sum())
    return result


def correspondence(rows, position_scale, velocity_scale):
    import torch
    maximum = torch.empty(23, 23)
    position_range = torch.empty(23, 23)
    velocity_maximum = torch.empty(23, 23)
    for index in range(23):
        maximum[index] = (rows["qpos"]-rows["amp"][:, index:index+1]).abs().max(0).values
        offset = rows["qpos"]-rows["critic_pos"][:, index:index+1]/position_scale[index]
        position_range[index] = offset.max(0).values-offset.min(0).values
        velocity_maximum[index] = (rows["qvel"]*velocity_scale[index]
                                   - rows["critic_vel"][:, index:index+1]).abs().max(0).values
    matches = maximum <= 1e-7
    if not bool((matches.sum(1) == 1).all()):
        raise ValueError("AMP/qpos mapping is not uniquely identified within 1e-7")
    order = maximum.argmin(1)
    if len(set(order.tolist())) != 23:
        raise ValueError("AMP/qpos correspondence is not a one-to-one permutation")
    if (not bool(((position_range <= 1e-6).sum(1) == 1).all())
            or not bool((position_range.argmin(1) == order).all())
            or not bool(((velocity_maximum <= 1e-7).sum(1) == 1).all())
            or not bool((velocity_maximum.argmin(1) == order).all())):
        raise ValueError("Critic joint coordinates do not have the same unique robot ordering")
    def wrong_margin(matrix):
        other = matrix.clone()
        other[torch.arange(23), order] = float("inf")
        return float(other.min())
    return order, {"amp_unique": True, "critic_position_unique": True, "critic_velocity_unique": True,
        "nonmatching_amp_min_max_abs": wrong_margin(maximum),
        "nonmatching_position_min_offset_range": wrong_margin(position_range),
        "nonmatching_velocity_min_max_abs": wrong_margin(velocity_maximum)}


def verify_formula(rows, order, default_pos, default_vel, position_scale, velocity_scale):
    import torch
    qpos, qvel = rows["qpos"][:, order], rows["qvel"][:, order]
    errors = {"amp_max_abs": float((rows["amp"]-qpos).abs().max()),
        "critic_position_max_abs": float((rows["critic_pos"]-(qpos-default_pos)*position_scale).abs().max()),
        "critic_velocity_max_abs": float((rows["critic_vel"]-(qvel-default_vel)*velocity_scale).abs().max())}
    if errors["amp_max_abs"] > 1e-7 or errors["critic_position_max_abs"] > 1e-6 or errors["critic_velocity_max_abs"] > 1e-7:
        raise ValueError("Source-derived joint formulas do not reproduce real replay: " + str(errors))
    return dict(errors, stored_state_rows=rows["stored_state_rows"], valid_next_rows=rows["valid_next_rows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("label must contain ASCII letters, digits, underscore or hyphen")
    import torch
    from audit_supermimic_model import projection_adapter, sha
    from mjlab_husky.supermimic.env_adapter import make_joint_projection_spec, joint_projection_spec_sha256
    torch.set_num_threads(1)
    started = time.perf_counter()
    output = args.out.resolve()/"data"/f"{args.label}_joint_metadata.json"
    if output.exists():
        raise FileExistsError("Preserve metadata evidence; choose a new label")
    corpus = torch.load(args.replay_corpus, map_location="cpu", weights_only=False)
    if corpus.get("format") != "husky-supermimic-real-replay-corpus-v1":
        raise ValueError("Unsupported complete real replay format")
    n, state_dim, _, _, _ = corpus["replay_state_dict"]["shape"]
    split = min(max(1, int(.8*n)), n-1)
    train_ids, heldout_ids = corpus["train_environment_ids"], corpus["heldout_environment_ids"]
    if n < 2 or train_ids != list(range(split)) or heldout_ids != list(range(split, n)):
        raise ValueError("Require original disjoint first80% train/last20% held-out split")
    root = Path(__file__).resolve().parents[1]
    names, defaults, files = source_joint_defaults(root)
    groups = corpus["physics_groups"]
    qadr = torch.tensor(groups["robot_joint_angle"]["indices"], dtype=torch.long)
    vadr = torch.tensor(groups["robot_joint_velocity"]["indices"], dtype=torch.long)
    if len(qadr) != 23 or len(vadr) != 23:
        raise ValueError("Source XML and named robot joint coordinate groups differ")
    preliminary = make_joint_projection_spec(corpus["layout"], corpus["critic_terms"], names,
        qadr, vadr, defaults["joint_pos"], defaults["joint_vel"])
    ps = torch.tensor(preliminary["critic_joint_pos_scale"])
    vs = torch.tensor(preliminary["critic_joint_vel_scale"])
    if bool((ps == 0).any()) or bool((vs == 0).any()):
        raise ValueError("Zero observation scale cannot establish unique physical identity")
    train = replay_rows(corpus, train_ids)
    order, evidence = correspondence(train, ps, vs)
    selected_names = [names[index] for index in order.tolist()]
    dp, dv = defaults["joint_pos"][order], defaults["joint_vel"][order]
    train_evidence = verify_formula(train, order, dp, dv, ps, vs)
    del train
    held = replay_rows(corpus, heldout_ids)
    held_evidence = verify_formula(held, order, dp, dv, ps, vs)
    del held
    spec = make_joint_projection_spec(corpus["layout"], corpus["critic_terms"], selected_names,
        qadr[order], vadr[order], dp, dv)
    if spec["state_dim"] != state_dim:
        raise ValueError("Replay state width and metadata layout differ")
    wrapper = {"spec": spec, "spec_sha256": joint_projection_spec_sha256(spec), "provenance": {
        "source": "G1 XML natural joint order/config default pose, verified by unique train-only real-replay correspondence",
        "corpus_path": str(args.replay_corpus.resolve()), "corpus_sha256": sha(args.replay_corpus),
        "saved_audit_batch_sha256": corpus["metadata"]["saved_audit_batch_sha256"],
        "source_sha256": {str(path.relative_to(root)): sha(path) for path in files},
        "adapter_source_sha256": sha(root/"src/mjlab_husky/supermimic/env_adapter.py"),
        "tool_sha256": sha(__file__), "train_environment_ids": train_ids, "heldout_environment_ids": heldout_ids,
        "uniqueness_from_training_only": evidence, "training_formula_validation": train_evidence,
        "heldout_formula_validation": held_evidence, "elapsed_seconds": time.perf_counter()-started,
        "scope": "No learned dynamics, future physical trajectories or held-out fitting parameters enter the constant spec. Held-out data verifies the fixed train/source mapping only; no optimizer or simulator runs."}}
    # Exercise the same strict setter used by offline model audits.
    projection_adapter(corpus, corpus["metadata"]["collection"], wrapper)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(wrapper, indent=2, allow_nan=False))
    print(json.dumps({"metadata": str(output), "metadata_sha256": sha(output),
        "spec_sha256": wrapper["spec_sha256"], "training": train_evidence,
        "heldout": held_evidence, "uniqueness": evidence}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
