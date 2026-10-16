"""One-block migration smoke test for the real WAM-to-RMBench path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from vista_wam.control.planner_guard import install_guard
from vista_wam.control.robotwin_contract import decode_raw, observed_raw
from vista_wam.evaluation.put_back import MAT_NAMES, run_auxiliary_prefix, target_score
from vista_wam.evaluation.target_interface import instruction
from vista_wam.model.client import request
from vista_wam.sim.continuation import install_continuation
from vista_wam.sim.environment import create_env, save_observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18774)
    parser.add_argument("--seed", type=int, default=100065)
    parser.add_argument("--target", choices=MAT_NAMES, default="front")
    parser.add_argument("--template", choices=("suffix", "explicit"), default="explicit")
    parser.add_argument("--max-actions", type=int, default=16, choices=range(1, 17))
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "test_scope": "one native WAM action block",
              "task_success_evaluated": False, "seed": args.seed, "target": args.target,
              "max_wam_actions": args.max_actions, "new_training_steps": 0}
    started = time.perf_counter()
    import urllib.request
    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}", timeout=30) as response:
        metadata = json.load(response)
    result["model_identity"] = metadata
    if metadata.get("release") != "robbyant/lingbot-va-posttrain-robotwin":
        raise RuntimeError("Wrong model release")
    if metadata.get("trainable_parameters") != 0 or metadata.get("memorywam_enabled") is not False:
        raise RuntimeError("Model is not the frozen external-memory baseline")
    env = create_env(args.seed, out, task="put_back_block", camera="D435")
    reset = save_observation(env, out / "reset_initial.npz")
    result["camera_shapes"] = {name: list(reset["observation"][name]["rgb"].shape)
                               for name in ("head_camera", "left_camera", "right_camera")}
    result["prefix"] = run_auxiliary_prefix(env, out)
    result["continuation"] = install_continuation(env, out)
    guards = install_guard(env)
    suffix = save_observation(env, out / "suffix_initial.npz")
    origin = observed_raw(suffix)
    prompt = instruction(args.template, args.target)
    reset_receipt = request("reset", port=args.port, prompt=prompt,
                            episode_id=out.name, full_reset=True)
    generated = request("infer_initial", port=args.port,
                        observation_npz=str(out / "suffix_initial.npz"))
    action = np.asarray(generated.pop("action"), dtype=float)
    if action.shape != (16, 2, 16) or not np.isfinite(action).all():
        raise RuntimeError(f"Unexpected WAM action shape: {action.shape}")
    np.savez(out / "generated_block.npz", relative_eef=action)
    sequence = action[:, 1, :].T[:args.max_actions]
    frames = []
    with (out / "actual_actions.jsonl").open("w", encoding="utf-8") as stream:
        for index, relative in enumerate(sequence, 1):
            command = decode_raw(relative, origin); before = int(env.take_action_cnt)
            env.take_action(command, action_type="ee")
            if int(env.take_action_cnt) != before + 1:
                raise RuntimeError("Decoded WAM command was not accepted exactly once")
            row = {"step": index, "source": "frozen posttrain-robotwin",
                   "command": command.tolist(), "measured": observed_raw(env.get_obs()).tolist(),
                   "score_NOT_policy_input": target_score(env, args.target)}
            stream.write(json.dumps(row) + "\n")
            if index % 4 == 0:
                file = out / f"observation_{index:04d}.npz"; save_observation(env, file); frames.append(file)
    commit = None
    if args.max_actions == 16:
        commit = request("commit", port=args.port, state_npz=str(out / "generated_block.npz"),
                         observations=[str(p) for p in frames], history_source="commanded_DIAGNOSTIC")
    save_observation(env, out / "final.npz")
    result.update(status="passed", instruction=prompt, reset_receipt=reset_receipt,
                  generated_shape=list(action.shape), finite_actions=True,
                  wam_commands_accepted=args.max_actions, real_feedback_frames=len(frames),
                  commit_receipt=commit, final_score_NOT_acceptance_criterion=target_score(env,args.target),
                  planner_guard_rejections={arm: guard.rejections for arm, guard in guards.items()},
                  migration_acceptance={"correct_release": True, "frozen_weights": True,
                                        "three_cameras": all(v == [240,320,3] for v in result["camera_shapes"].values()),
                                        "native_action_tensor": True,
                                        "decoded_commands_executed": args.max_actions,
                                        "real_history_committed": args.max_actions == 16})
    result["wall_seconds"] = time.perf_counter() - started
    (out / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["migration_acceptance"], indent=2))


if __name__ == "__main__":
    main()

