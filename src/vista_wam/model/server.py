"""Serve the frozen posttrain-robotwin release on an isolated loopback endpoint.

Inputs: explicit reset/inference/commit operations and project-confined real
camera NPZ files. Outputs: native action tensors plus cache/identity receipts.
This preserves the phase-eight inference contract while writing only phase-nine
service evidence; it does not modify weights or enable MemoryWAM.
"""
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import random
import sys
import time


from vista_wam.config import load_config, project_root

ROOT = project_root()


def main():
    """Load the frozen release once and expose bounded loopback operations."""
    import numpy as np
    import torch

    portable = load_config()
    config = {
        "repository": portable["third_party"]["lingbot_va"],
        "model_dir": portable["models"]["lingbot_posttrain_robotwin"],
        "release_identity": "robbyant/lingbot-va-posttrain-robotwin",
        "port": int(os.getenv("VISTA_WAM_PORT", portable["service"]["port"])),
        "video_steps": portable["service"]["video_steps"],
        "action_steps": portable["service"]["action_steps"],
        "seed": portable["service"]["seed"],
        "new_training_steps": 0,
    }
    sys.path.insert(0, config["repository"])
    from wan_va.configs import VA_CONFIGS
    from wan_va.wan_va_server import VA_Server

    cfg = copy.deepcopy(VA_CONFIGS["robotwin"])
    cfg.wan22_pretrained_model_name_or_path = config["model_dir"]
    cfg.enable_offload = True
    cfg.fsdp_cpu_offload = False
    cfg.rank = cfg.local_rank = 0
    cfg.world_size = 1
    cfg.save_root = str(ROOT / "outputs/model_debug")
    cfg.enable_wandb = False
    cfg.num_inference_steps = config["video_steps"]
    cfg.action_num_inference_steps = config["action_steps"]
    torch.set_num_threads(8)
    started = time.perf_counter()
    model = VA_Server(cfg)
    model.text_encoder.float()
    for module in [model.transformer, model.text_encoder, model.vae, model.streaming_vae_half.vae]:
        module.eval().requires_grad_(False)
    if model.memorywam_enabled:
        raise RuntimeError("Frozen release baseline cannot enable MemoryWAM")

    episode = {"initial": None, "blocks": [], "prompt": None, "id": None, "full_reset": True}

    def confined(value):
        """Require an existing regular input file under this project."""
        path = Path(value).resolve()
        if not path.is_file():
            raise ValueError("Input must be an existing regular file")
        return path

    def observation(value):
        """Map saved actual NPZ cameras to the released RoboTwin input schema."""
        with np.load(confined(value)) as data:
            return dict(
                zip(
                    [
                        "observation.images.cam_high",
                        "observation.images.cam_left_wrist",
                        "observation.images.cam_right_wrist",
                    ],
                    [data[key] for key in ["head_camera", "left_camera", "right_camera"]],
                ),
                **{"observation.state": data["state"], "task": episode["prompt"]},
            )

    def receipt():
        """Describe actual stream caches and current episode conditioning."""
        return {
            "episode_id": episode["id"],
            "prompt": episode["prompt"],
            "rng_seed": config["seed"],
            "frame_st_id": model.frame_st_id,
            "blocks": len(episode["blocks"]),
            "full_reset": episode["full_reset"],
            "half_cache_nonempty": sum(value is not None for value in model.streaming_vae_half.feat_cache),
            "high_cache_nonempty": sum(value is not None for value in model.streaming_vae.feat_cache),
            "prompt_sha256": None if model.prompt_embeds is None else hashlib.sha256(
                model.prompt_embeds.float().cpu().numpy().tobytes()
            ).hexdigest(),
        }

    def reset(prompt, identifier=None):
        """Reset all model/RNG state for an independent paired assignment."""
        if not isinstance(prompt, str) or not prompt or len(prompt) > 180 or "/" in prompt:
            raise ValueError("Invalid prompt")
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        model.infer({"reset": True, "prompt": prompt})
        if episode["full_reset"]:
            model.streaming_vae_half.clear_cache()
        episode.update(prompt=prompt, id=identifier or episode["id"])

    def commit(block):
        """Commit one complete action/real-image block with labeled provenance."""
        with np.load(confined(block["state_npz"])) as data:
            state = data["relative_eef"]
        if state.shape != (16, 2, 16) or not np.isfinite(state).all():
            raise ValueError("Bad history tensor")
        if len(block["observations"]) != (4 if model.frame_st_id == 0 else 8):
            raise ValueError("Bad history alignment")
        if block["history_source"] not in [
            "measured_executed",
            "measured_with_native_bootstrap",
            "commanded_DIAGNOSTIC",
        ]:
            raise ValueError("Unlabeled history")
        model.infer({
            "obs": [observation(path) for path in block["observations"]],
            "state": state,
            "compute_kv_cache": True,
        })

    class Handler(BaseHTTPRequestHandler):
        """Serve serial experiment operations and return structured failures."""

        def respond(self, code, data):
            """Write one JSON response."""
            raw = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            """Return service identity and current episode state."""
            self.respond(200, dict(metadata, episode=receipt() if episode["prompt"] else None))

        def do_POST(self):
            """Dispatch reset, native first inference, inference, and commit."""
            try:
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                operation = data["operation"]
                call_started = time.perf_counter()
                with torch.inference_mode():
                    if operation == "reset":
                        episode.update(initial=None, blocks=[], full_reset=data.get("full_reset", True))
                        reset(data["prompt"], data["episode_id"])
                        result = receipt()
                    elif operation == "infer_initial":
                        if model.frame_st_id != 0 or episode["initial"] is not None:
                            raise ValueError("Native first inference requires fresh reset")
                        episode["initial"] = str(confined(data["observation_npz"]))
                        response = model.infer({
                            "obs": observation(episode["initial"]),
                            "prompt": episode["prompt"],
                        })
                        result = {"action": np.asarray(response["action"]).tolist(), "condition": receipt()}
                    elif operation == "infer":
                        response = model.infer({
                            "obs": observation(episode["initial"]),
                            "prompt": episode["prompt"],
                        })
                        result = {"action": np.asarray(response["action"]).tolist(), "condition": receipt()}
                    elif operation == "commit":
                        block = {key: data[key] for key in ["observations", "state_npz", "history_source"]}
                        commit(block)
                        episode["blocks"].append(block)
                        result = receipt()
                    else:
                        raise ValueError("Unknown phase-nine operation")
                result["seconds"] = time.perf_counter() - call_started
                self.respond(200, result)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.respond(500, {"error": str(exc), "error_type": type(exc).__name__})

    metadata = {
        "status": "ready",
        "protocol": "portable-v1",
        "video_guidance_scale": cfg.guidance_scale,
        "action_guidance_scale": cfg.action_guidance_scale,
        "port": config["port"],
        "pid": os.getpid(),
        "trainable_parameters": 0,
        "memorywam_enabled": False,
        "release": config["release_identity"],
        "load_seconds": time.perf_counter() - started,
        "config": config,
    }
    destination = ROOT / "outputs/service"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    HTTPServer(("127.0.0.1", config["port"]), Handler).serve_forever()


if __name__ == "__main__":
    main()
