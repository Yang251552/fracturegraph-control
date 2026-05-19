#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import deep_update, load_config


Variant = Dict[str, Any]


THRESHOLD_VARIANTS: List[Variant] = [
    {"name": "checkpoint_threshold", "break_threshold": None, "planner": {}},
    {"name": "break_th_085", "break_threshold": 0.85, "planner": {}},
    {"name": "break_th_090", "break_threshold": 0.90, "planner": {}},
    {"name": "break_th_095", "break_threshold": 0.95, "planner": {}},
]

PLANNER_VARIANTS: List[Variant] = [
    {"name": "break_th_090_budget", "break_threshold": 0.90, "planner": {"population": 384, "iterations": 6}},
    {
        "name": "break_th_090_lighter_cost",
        "break_threshold": 0.90,
        "planner": {"objective": {"action_cost_weight": 0.015}},
    },
    {
        "name": "break_th_090_lighter_overbreak",
        "break_threshold": 0.90,
        "planner": {"objective": {"action_cost_weight": 0.015, "overbreak_penalty": 0.4}},
    },
    {
        "name": "break_th_090_h6_budget",
        "break_threshold": 0.90,
        "planner": {"horizon": 6, "population": 384, "iterations": 6},
    },
]

VERIFY_VARIANTS: List[Variant] = [
    {"name": "checkpoint_threshold", "break_threshold": None, "planner": {}},
    {"name": "break_th_090", "break_threshold": 0.90, "planner": {}},
    {
        "name": "break_th_090_h6_budget",
        "break_threshold": 0.90,
        "planner": {"horizon": 6, "population": 384, "iterations": 6},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep surrogate rollout thresholds and planner settings.")
    parser.add_argument("--experiment", default="configs/experiment/reshape_targets_balanced.yaml")
    parser.add_argument("--checkpoint", default="results/checkpoints/balanced_monitor_20260518_213730/surrogate_best.pt")
    parser.add_argument("--output-root")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--suite", choices=["threshold", "planner", "full", "verify"], default="full")
    parser.add_argument(
        "--targets",
        default="",
        help="Optional comma-separated target filter, for example rectangle_cut,circular_notch.",
    )
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--compare", default="cem_surrogate")
    parser.add_argument("--surrogate-node-threshold", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="fracturegraph-control")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_root = ROOT / (args.output_root or f"results/rollouts/planner_sweep_{run_id}")
    config_root = output_root / "_configs"
    output_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)

    exp = load_config(ROOT / args.experiment)
    planner_config = load_config(ROOT / str(exp["planner_config"]))
    target_filter = _target_filter(args.targets)
    variants = _variants(args.suite)
    combined_rows: List[Dict[str, Any]] = []
    wandb_run = _init_wandb(args, output_root, target_filter)

    for variant_step, variant in enumerate(variants):
        variant_name = str(variant["name"])
        variant_output = output_root / variant_name
        summary_path = variant_output / "eval_summary.csv"
        if args.skip_existing and summary_path.exists():
            print(f"sweep: reusing {variant_name}", flush=True)
        else:
            variant_exp = _variant_experiment(exp, target_filter, args.compare)
            variant_planner = deep_update(copy.deepcopy(planner_config), variant.get("planner", {}))
            planner_path = config_root / f"{variant_name}.planner.yaml"
            experiment_path = config_root / f"{variant_name}.experiment.yaml"
            _write_yaml(planner_path, variant_planner)
            variant_exp["planner_config"] = _rel(planner_path)
            _write_yaml(experiment_path, variant_exp)
            cmd = [
                args.python,
                "scripts/eval_rollout.py",
                "--experiment",
                _rel(experiment_path),
                "--checkpoint",
                args.checkpoint,
                "--output-dir",
                _rel(variant_output),
            ]
            if variant.get("break_threshold") is not None:
                cmd.extend(["--surrogate-break-threshold", f"{float(variant['break_threshold']):.2f}"])
            if args.surrogate_node_threshold is not None:
                cmd.extend(["--surrogate-node-threshold", f"{args.surrogate_node_threshold:.2f}"])
            if args.max_seeds is not None:
                cmd.extend(["--max-seeds", str(args.max_seeds)])
            if args.device:
                cmd.extend(["--device", args.device])
            print(f"sweep: running {variant_name}", flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)

        rows = _read_csv(summary_path)
        for row in rows:
            annotated = _annotate_row(row, variant, planner_config, variant_output, variant_step)
            combined_rows.append(annotated)
            _log_wandb_row(wandb_run, annotated)
        _write_outputs(output_root, combined_rows)

    _print_recommendation(combined_rows)
    if wandb_run is not None:
        wandb_run.summary["completed"] = True
        wandb_run.summary["rows"] = len(combined_rows)
        wandb_run.finish()
    print(f"sweep: wrote summary to {output_root / 'sweep_summary.csv'}", flush=True)


def _variants(suite: str) -> List[Variant]:
    if suite == "threshold":
        return THRESHOLD_VARIANTS
    if suite == "planner":
        return THRESHOLD_VARIANTS[:1] + PLANNER_VARIANTS
    if suite == "verify":
        return VERIFY_VARIANTS
    return THRESHOLD_VARIANTS + PLANNER_VARIANTS


def _target_filter(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _variant_experiment(exp: Dict[str, Any], target_filter: List[str], compare: str) -> Dict[str, Any]:
    out = copy.deepcopy(exp)
    if target_filter:
        out["targets"] = [target for target in out["targets"] if str(target["type"]) in target_filter]
    out.setdefault("eval", {})["compare"] = [item.strip() for item in compare.split(",") if item.strip()]
    return out


def _annotate_row(
    row: Dict[str, str],
    variant: Variant,
    base_planner: Dict[str, Any],
    output_dir: Path,
    variant_step: int,
) -> Dict[str, Any]:
    planner = deep_update(copy.deepcopy(base_planner), variant.get("planner", {}))
    objective = planner.get("objective", {})
    return {
        "variant": variant["name"],
        "variant_step": variant_step,
        "break_threshold": "checkpoint" if variant.get("break_threshold") is None else f"{float(variant['break_threshold']):.2f}",
        "horizon": planner.get("horizon"),
        "population": planner.get("population"),
        "iterations": planner.get("iterations"),
        "action_cost_weight": objective.get("action_cost_weight"),
        "overbreak_penalty": objective.get("overbreak_penalty"),
        **row,
        "output_dir": _rel(output_dir),
        "report": _rel(output_dir / "report.html"),
    }


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_outputs(output_root: Path, rows: List[Dict[str, Any]]) -> None:
    _write_csv(output_root / "sweep_summary.csv", rows)
    (output_root / "sweep_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_recommendation(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    scored = sorted(rows, key=lambda row: float(row["mean_final_iou"]), reverse=True)
    print("sweep: best rows by mean_final_iou", flush=True)
    for row in scored[:6]:
        print(
            "  "
            f"{row['target']} {row['variant']} "
            f"iou={float(row['mean_final_iou']):.4f} "
            f"undercut={float(row.get('mean_undercut', 0.0)):.4f} "
            f"overbreak={float(row.get('mean_overbreak', 0.0)):.4f}",
            flush=True,
        )


def _init_wandb(args: argparse.Namespace, output_root: Path, target_filter: List[str]):
    if not args.wandb:
        return None
    try:
        import wandb
    except ModuleNotFoundError:
        print("sweep: W&B requested but wandb is not installed; continuing without W&B.", flush=True)
        return None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or output_root.name,
        job_type="planner_sweep",
        config={
            "suite": args.suite,
            "targets": target_filter,
            "checkpoint": args.checkpoint,
            "output_root": _rel(output_root),
            "compare": args.compare,
            "device": args.device,
        },
    )
    run.define_metric("variant_step")
    run.define_metric("rollout/*", step_metric="variant_step")
    return run


def _log_wandb_row(wandb_run, row: Dict[str, Any]) -> None:
    if wandb_run is None:
        return
    payload: Dict[str, Any] = {
        "variant": row["variant"],
        "target": row["target"],
        "variant_step": int(row["variant_step"]),
        "break_threshold": row["break_threshold"],
        "rollout/mean_final_iou": float(row["mean_final_iou"]),
        "rollout/best_final_iou": float(row["best_final_iou"]),
        "rollout/mean_score": float(row["mean_score"]),
        "rollout/mean_force": float(row["mean_force"]),
        "rollout/mean_undercut": float(row.get("mean_undercut", 0.0)),
        "rollout/mean_overbreak": float(row.get("mean_overbreak", 0.0)),
        "rollout/horizon": int(row["horizon"]),
        "rollout/population": int(row["population"]),
        "rollout/iterations": int(row["iterations"]),
        "rollout/action_cost_weight": float(row["action_cost_weight"]),
        "rollout/overbreak_penalty": float(row["overbreak_penalty"]),
    }
    target = str(row["target"])
    payload[f"rollout_by_target/{target}/mean_final_iou"] = float(row["mean_final_iou"])
    payload[f"rollout_by_target/{target}/mean_undercut"] = float(row.get("mean_undercut", 0.0))
    payload[f"rollout_by_target/{target}/mean_overbreak"] = float(row.get("mean_overbreak", 0.0))
    wandb_run.log(payload)


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


if __name__ == "__main__":
    main()
