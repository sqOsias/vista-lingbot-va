"""Download required weights without adding them to Git."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


MODELS = {
    "lingbot": "robbyant/lingbot-va-posttrain-robotwin",
    "qwen": "Qwen/Qwen3-VL-8B-Instruct",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=MODELS)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "checkpoints")
    args = parser.parse_args(); args.root.mkdir(parents=True, exist_ok=True)
    names = {"lingbot": "lingbot-va-posttrain-robotwin", "qwen": "Qwen3-VL-8B-Instruct"}
    for key in args.models:
        destination = args.root / names[key]
        snapshot_download(repo_id=MODELS[key], local_dir=destination)
        print(f"{MODELS[key]} -> {destination}")


if __name__ == "__main__": main()

