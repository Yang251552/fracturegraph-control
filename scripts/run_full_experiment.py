#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]


PRESETS = {
    "dev": {
        "data": "configs/data/lattice_16_dev.yaml",
        "model": "configs/model/meshgnn_small.yaml",
        "train": "configs/train/surrogate_dev.yaml",
        "experiment": "configs/experiment/reshape_targets_dev.yaml",
        "checkpoint": "results/checkpoints/dev/surrogate_best.pt",
        "output": "results/rollouts/dev",
    },
    "formal": {
        "data": "configs/data/lattice_32.yaml",
        "model": "configs/model/meshgnn_small.yaml",
        "train": "configs/train/surrogate.yaml",
        "experiment": "configs/experiment/reshape_targets.yaml",
        "checkpoint": "results/checkpoints/surrogate_best.pt",
        "output": "results/rollouts/formal",
    },
    "balanced": {
        "data": "configs/data/lattice_24_balanced.yaml",
        "model": "configs/model/meshgnn_balanced.yaml",
        "train": "configs/train/surrogate_balanced.yaml",
        "experiment": "configs/experiment/reshape_targets_balanced.yaml",
        "checkpoint": "results/checkpoints/balanced/surrogate_best.pt",
        "output": "results/rollouts/balanced",
    },
    "aws48": {
        "data": "configs/data/lattice_48.yaml",
        "model": "configs/model/meshgnn_medium.yaml",
        "train": "configs/train/surrogate.yaml",
        "experiment": "configs/experiment/reshape_targets_48.yaml",
        "checkpoint": "results/checkpoints/surrogate_best.pt",
        "output": "results/rollouts/aws48",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset generation, surrogate training, and planning evaluation.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="formal")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--max-seeds", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.preset]
    if not args.skip_data:
        _run(
            [
                args.python,
                "scripts/generate_dataset.py",
                "--config",
                preset["data"],
                "--render-preview",
            ]
        )
    if not args.skip_train:
        _run(
            [
                args.python,
                "scripts/train_surrogate.py",
                "--data-config",
                preset["data"],
                "--model-config",
                preset["model"],
                "--train-config",
                preset["train"],
            ]
        )
    if not args.skip_eval:
        cmd = [
            args.python,
            "scripts/eval_rollout.py",
            "--experiment",
            preset["experiment"],
            "--checkpoint",
            preset["checkpoint"],
            "--output-dir",
            preset["output"],
        ]
        if args.max_seeds is not None:
            cmd.extend(["--max-seeds", str(args.max_seeds)])
        _run(cmd)


def _run(cmd: List[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
