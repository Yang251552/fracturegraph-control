#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.lattice_sim import create_initial_state, target_mask
from src.models import build_model
from src.planner import SurrogateRollout, cem_plan, greedy_plan, random_plan, rollout_simulator
from src.viz import render_episode_svg, render_report_html, write_rollout_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fracture planning baselines across targets and seeds.")
    parser.add_argument("--experiment", default="configs/experiment/reshape_targets.yaml")
    parser.add_argument("--checkpoint", default="results/checkpoints/surrogate_best.pt")
    parser.add_argument("--output-dir", default="results/rollouts")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument(
        "--surrogate-break-threshold",
        type=float,
        help="Override the edge break probability threshold stored in the checkpoint.",
    )
    parser.add_argument(
        "--surrogate-node-threshold",
        type=float,
        help="Override the node alive probability threshold stored in the checkpoint.",
    )
    parser.add_argument("--device", default="auto", help="Device for surrogate rollout: auto, cpu, cuda, or cuda:0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp = load_config(ROOT / args.experiment)
    data_config = load_config(ROOT / exp["data_config"])
    planner_config = load_config(ROOT / exp["planner_config"])
    seed = int(args.seed if args.seed is not None else exp.get("seed", 0))
    compare = list(exp.get("eval", {}).get("compare", ["random", "greedy", "cem_surrogate", "cem_oracle"]))  # type: ignore[union-attr]
    target_specs = _target_specs(exp)
    seeds = _eval_seeds(exp, seed, args.max_seeds)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    surrogate_rollout = None
    if "cem_surrogate" in compare:
        surrogate_rollout = try_load_surrogate(
            ROOT / args.checkpoint,
            planner_config,
            break_threshold=args.surrogate_break_threshold,
            node_threshold=args.surrogate_node_threshold,
            device=args.device,
        )
        if surrogate_rollout is None:
            compare = [name for name in compare if name != "cem_surrogate"]

    rows: List[Dict[str, object]] = []
    figures: List[str] = []
    for target_spec in target_specs:
        target = target_mask(int(target_spec["size"]), str(target_spec["type"]), int(target_spec.get("margin", 4)))
        for eval_seed in seeds:
            initial = create_initial_state(data_config, random.Random(eval_seed))
            for method_idx, method in enumerate(compare):
                method_rng = random.Random(eval_seed + 1000 * method_idx + _target_hash(str(target_spec["type"])))
                if method == "random":
                    result = random_plan(initial, target, data_config, planner_config, method_rng)
                elif method == "greedy":
                    result = greedy_plan(initial, target, data_config, planner_config, method_rng)
                elif method == "cem_oracle":
                    result = cem_plan(
                        initial,
                        target,
                        data_config,
                        planner_config,
                        method_rng,
                        rollout=lambda state, actions, r=method_rng: rollout_simulator(state, actions, data_config, r),
                        method="cem_oracle",
                    )
                elif method == "cem_surrogate" and surrogate_rollout is not None:
                    result = cem_plan(initial, target, data_config, planner_config, method_rng, surrogate_rollout, "cem_surrogate")
                else:
                    continue

                target_name = str(target_spec["type"])
                basename = f"{target_name}.seed_{eval_seed}.{result.method}"
                row = {
                    "target": target_name,
                    "seed": eval_seed,
                    "method": result.method,
                    "final_iou": round(result.final_iou, 4),
                    "score": round(result.score, 4),
                    "actions": len(result.actions),
                    "mean_force": round(sum(action.force for action in result.actions) / max(1, len(result.actions)), 4),
                }
                row.update(_shape_diagnostics(result.states[-1].alive_mask(), target))
                rows.append(row)
                fig_path = output_dir / f"{basename}.svg"
                json_path = output_dir / f"{basename}.json"
                render_episode_svg(fig_path, result.states, target, result.actions, f"{target_name} / {result.method}")
                write_rollout_json(
                    json_path,
                    result.method,
                    result.states,
                    target,
                    result.actions,
                    {"score": result.score, "final_iou": result.final_iou},
                )
                figures.append(fig_path.relative_to(output_dir).as_posix())

    summary = _summarize(rows)
    (output_dir / "eval_metrics.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    _write_csv(output_dir / "eval_metrics.csv", rows)
    _write_csv(output_dir / "eval_summary.csv", summary)
    render_report_html(output_dir / "report.html", summary, figures)
    print(json.dumps(summary, indent=2))
    print(f"Wrote report to {output_dir / 'report.html'}")


def _target_specs(exp: Dict[str, object]) -> List[Dict[str, object]]:
    if "targets" in exp:
        return [dict(item) for item in exp["targets"]]  # type: ignore[index]
    target = dict(exp["target"])  # type: ignore[index]
    num_targets = int(exp.get("eval", {}).get("num_targets", 1))  # type: ignore[union-attr]
    return [target for _ in range(num_targets)]


def _eval_seeds(exp: Dict[str, object], seed: int, max_seeds: Optional[int]) -> List[int]:
    eval_cfg = exp.get("eval", {})
    if isinstance(eval_cfg, dict) and "seeds" in eval_cfg:
        seeds = [int(value) for value in eval_cfg["seeds"]]  # type: ignore[index]
    else:
        n = int(eval_cfg.get("num_seeds", 1)) if isinstance(eval_cfg, dict) else 1
        seeds = [seed + i for i in range(n)]
    if max_seeds is not None:
        requested = max(1, max_seeds)
        if len(seeds) < requested:
            next_seed = seed
            existing = set(seeds)
            while len(seeds) < requested:
                if next_seed not in existing:
                    seeds.append(next_seed)
                    existing.add(next_seed)
                next_seed += 1
        seeds = seeds[:requested]
    return seeds


def _summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((row["target"], row["method"]), []).append(row)
    summary: List[Dict[str, object]] = []
    for (target, method), group in sorted(groups.items()):
        ious = [float(row["final_iou"]) for row in group]
        scores = [float(row["score"]) for row in group]
        forces = [float(row["mean_force"]) for row in group]
        undercuts = [float(row.get("undercut", 0.0)) for row in group]
        overbreaks = [float(row.get("overbreak", 0.0)) for row in group]
        summary.append(
            {
                "target": target,
                "method": method,
                "runs": len(group),
                "mean_final_iou": round(sum(ious) / len(ious), 4),
                "best_final_iou": round(max(ious), 4),
                "mean_score": round(sum(scores) / len(scores), 4),
                "mean_force": round(sum(forces) / len(forces), 4),
                "mean_undercut": round(sum(undercuts) / len(undercuts), 4),
                "mean_overbreak": round(sum(overbreaks) / len(overbreaks), 4),
            }
        )
    return summary


def _shape_diagnostics(alive: List[int], target: List[int]) -> Dict[str, float]:
    remove_total = 0
    undercut = 0
    keep_total = 0
    overbreak = 0
    for is_alive, should_keep in zip(alive, target):
        if bool(should_keep):
            keep_total += 1
            if not bool(is_alive):
                overbreak += 1
        else:
            remove_total += 1
            if bool(is_alive):
                undercut += 1
    return {
        "undercut": round(undercut / max(1, remove_total), 4),
        "overbreak": round(overbreak / max(1, keep_total), 4),
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _target_hash(name: str) -> int:
    return sum((i + 1) * ord(char) for i, char in enumerate(name))


def try_load_surrogate(
    checkpoint: Path,
    planner_config,
    break_threshold: Optional[float] = None,
    node_threshold: Optional[float] = None,
    device: str = "auto",
):
    if not checkpoint.exists():
        print(f"Skipping cem_surrogate because checkpoint is missing: {checkpoint}")
        return None
    try:
        import torch
    except ModuleNotFoundError:
        print("Skipping cem_surrogate because PyTorch is not installed.")
        return None
    payload = torch.load(checkpoint, map_location="cpu")
    model = build_model(payload["model_config"])
    model.load_state_dict(payload["model_state"])
    resolved_device = _resolve_device(device, torch)
    model.to(resolved_device)
    metrics = payload.get("metrics", {})
    threshold = float(
        break_threshold
        if break_threshold is not None
        else metrics.get("edge_prob_threshold", planner_config.get("surrogate_break_threshold", 0.45))
    )
    node_alive_threshold = float(
        node_threshold
        if node_threshold is not None
        else metrics.get("node_prob_threshold", planner_config.get("surrogate_node_threshold", 0.5))
    )
    print(
        "Loaded surrogate thresholds: "
        f"edge_break={threshold:.2f}, node_alive={node_alive_threshold:.2f}, device={resolved_device}",
        flush=True,
    )
    return SurrogateRollout(model, threshold=threshold, node_threshold=node_alive_threshold, device=resolved_device)


def _resolve_device(requested: str, torch) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        return "cpu"
    return requested


if __name__ == "__main__":
    main()
