#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.lattice_sim import create_initial_state, target_mask
from src.metrics import shape_iou
from src.models import build_model
from src.planner import SurrogateRollout, cem_plan, greedy_plan, random_plan, rollout_simulator
from src.viz import render_episode_svg, write_rollout_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan an impact sequence for a target mask.")
    parser.add_argument("--experiment", default="configs/experiment/reshape_targets.yaml")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--method", default="cem_oracle", choices=["random", "greedy", "cem_oracle", "cem_surrogate"])
    parser.add_argument("--checkpoint", default="results/checkpoints/surrogate_best.pt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-prefix", default="results/rollouts/plan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp = load_config(ROOT / args.experiment)
    data_config = load_config(ROOT / exp["data_config"])
    planner_config = load_config(ROOT / exp["planner_config"])
    seed = int(args.seed if args.seed is not None else exp.get("seed", 0))
    rng = random.Random(seed)

    initial = create_initial_state(data_config, rng)
    target_cfg = _select_target(exp, args.target_index)
    target = target_mask(int(target_cfg["size"]), str(target_cfg["type"]), int(target_cfg.get("margin", 4)))

    if args.method == "random":
        result = random_plan(initial, target, data_config, planner_config, rng)
    elif args.method == "greedy":
        result = greedy_plan(initial, target, data_config, planner_config, rng)
    elif args.method == "cem_oracle":
        result = cem_plan(
            initial,
            target,
            data_config,
            planner_config,
            rng,
            rollout=lambda state, actions: rollout_simulator(state, actions, data_config, rng),
            method="cem_oracle",
        )
    else:
        rollout = load_surrogate_rollout(ROOT / args.checkpoint, planner_config)
        result = cem_plan(initial, target, data_config, planner_config, rng, rollout=rollout, method="cem_surrogate")

    prefix = ROOT / args.output_prefix
    svg_path = prefix.with_suffix(f".{args.method}.svg")
    json_path = prefix.with_suffix(f".{args.method}.json")
    metrics = {"score": result.score, "final_iou": result.final_iou}
    render_episode_svg(svg_path, result.states, target, result.actions, f"{result.method} fracture plan")
    write_rollout_json(json_path, result.method, result.states, target, result.actions, metrics)
    print(f"{result.method}: final_iou={shape_iou(result.states[-1].alive_mask(), target):.3f} score={result.score:.3f}")
    print(f"Wrote {svg_path} and {json_path}")


def load_surrogate_rollout(checkpoint: Path, planner_config):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("CEM surrogate planning needs PyTorch and a trained checkpoint.") from exc
    if not checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu")
    model = build_model(payload["model_config"])
    model.load_state_dict(payload["model_state"])
    metrics = payload.get("metrics", {})
    threshold = float(metrics.get("edge_prob_threshold", planner_config.get("surrogate_break_threshold", 0.45)))
    node_threshold = float(metrics.get("node_prob_threshold", planner_config.get("surrogate_node_threshold", 0.5)))
    return SurrogateRollout(model, threshold=threshold, node_threshold=node_threshold, device="cpu")


def _select_target(exp, target_index: int):
    if "targets" in exp:
        targets = exp["targets"]
        return targets[max(0, min(target_index, len(targets) - 1))]
    return exp["target"]


if __name__ == "__main__":
    main()
