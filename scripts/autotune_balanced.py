#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config


METRIC_RE = re.compile(
    r"epoch=(?P<epoch>\d+)\s+"
    r"train_loss=(?P<train_loss>[-+0-9.eE]+)\s+"
    r"edge_bce=(?P<edge_bce>[-+0-9.eE]+)\s+"
    r"node_bce=(?P<node_bce>[-+0-9.eE]+)\s+"
    r"edge_f1=(?P<edge_f1>[-+0-9.eE]+)\s+"
    r"node_iou=(?P<node_iou>[-+0-9.eE]+)\s+"
    r"precision=(?P<precision>[-+0-9.eE]+)\s+"
    r"recall=(?P<recall>[-+0-9.eE]+)\s+"
    r"edge_threshold=(?P<edge_prob_threshold>[-+0-9.eE]+)"
)
WANDB_RUN_RE = re.compile(r"https://wandb\.ai/(?P<entity>[^/\s]+)/(?P<project>[^/\s]+)/runs/(?P<run_id>[A-Za-z0-9_-]+)")


@dataclass
class Decision:
    stop: bool
    reason: str
    detail: str


@dataclass
class Recipe:
    lr: float
    edge_bce_weight: float
    node_bce_weight: float
    dropout: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "lr": self.lr,
            "edge_bce_weight": self.edge_bce_weight,
            "node_bce_weight": self.node_bce_weight,
            "dropout": self.dropout,
        }


class Monitor:
    def __init__(self, warmup_epochs: int, patience: int, min_delta: float, min_edge_f1: float):
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.min_edge_f1 = min_edge_f1
        self.best_selection = -1.0
        self.best_epoch = 0
        self.metrics: List[Dict[str, float]] = []

    def update(self, metrics: Dict[str, float]) -> Decision:
        epoch = int(metrics["epoch"])
        selection = _selection_score(metrics)
        metrics["selection_score"] = selection
        self.metrics.append(metrics)
        if selection > self.best_selection + self.min_delta:
            self.best_selection = selection
            self.best_epoch = epoch

        print(
            "autotune: "
            f"epoch={epoch} selection={selection:.4f} best={self.best_selection:.4f}@{self.best_epoch} "
            f"edge_f1={metrics['edge_f1']:.3f} precision={metrics['precision']:.3f} recall={metrics['recall']:.3f}",
            flush=True,
        )

        if epoch < self.warmup_epochs:
            return Decision(False, "warmup", f"waiting until epoch {self.warmup_epochs}")

        recent = self.metrics[-min(3, len(self.metrics)) :]
        if len(recent) >= 3 and all(row["edge_f1"] < self.min_edge_f1 for row in recent):
            return Decision(
                True,
                "low_edge_f1",
                f"edge_f1 stayed below {self.min_edge_f1:.3f} for {len(recent)} validation epochs",
            )
        if len(recent) >= 2 and all(row["precision"] < 0.68 and row["recall"] > 0.90 for row in recent):
            return Decision(True, "precision_low", "high recall with low precision indicates over-breaking")
        if len(recent) >= 2 and all(row["precision"] > 0.88 and row["recall"] < 0.70 for row in recent):
            return Decision(True, "recall_low", "high precision with low recall indicates under-breaking")
        if epoch - self.best_epoch >= self.patience:
            return Decision(True, "plateau", f"selection_score did not improve for {self.patience} epochs")
        return Decision(False, "continue", "metrics are within the allowed envelope")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a guarded balanced autotuning loop.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.003)
    parser.add_argument("--min-edge-f1", type=float, default=0.82)
    parser.add_argument("--data-config", default="configs/data/lattice_24_balanced.yaml")
    parser.add_argument("--base-model-config", default="configs/model/meshgnn_balanced.yaml")
    parser.add_argument("--base-train-config", default="configs/train/surrogate_balanced.yaml")
    parser.add_argument("--experiment", default="configs/experiment/reshape_targets_balanced.yaml")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--poll-wandb", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_root = ROOT / "results" / "autotune" / "balanced" / run_id
    config_root = ROOT / "configs" / "autotune" / "balanced" / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_data:
        ensure_data(args)

    base_train = load_config(ROOT / args.base_train_config)
    base_model = load_config(ROOT / args.base_model_config)
    recipe = Recipe(
        lr=float(base_train.get("lr", 4e-4)),
        edge_bce_weight=float(base_train.get("loss", {}).get("edge_bce_weight", 1.0)),  # type: ignore[union-attr]
        node_bce_weight=float(base_train.get("loss", {}).get("node_bce_weight", 0.35)),  # type: ignore[union-attr]
        dropout=float(base_model.get("dropout", 0.05)),
    )
    summary: Dict[str, Any] = {"run_id": run_id, "attempts": []}

    for attempt in range(1, args.max_attempts + 1):
        attempt_name = f"attempt_{attempt:02d}"
        train_config_path, model_config_path, checkpoint_path, eval_output = write_attempt_configs(
            config_root,
            output_root,
            attempt_name,
            base_train,
            base_model,
            recipe,
        )
        print(f"autotune: starting {attempt_name} with {recipe.as_dict()}", flush=True)
        result = run_training_attempt(args, attempt_name, train_config_path, model_config_path, output_root)
        summary["attempts"].append(result)
        (output_root / "autotune_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if result["status"] == "finished":
            run_eval(args, checkpoint_path, eval_output)
            print(f"autotune: finished successfully on {attempt_name}", flush=True)
            break
        if attempt == args.max_attempts:
            print("autotune: reached max attempts without a completed training run", flush=True)
            break
        recipe = propose_next_recipe(recipe, result)
        print(f"autotune: next recipe -> {recipe.as_dict()}", flush=True)


def ensure_data(args: argparse.Namespace) -> None:
    data_config = load_config(ROOT / args.data_config)
    out_dir = ROOT / str(data_config["output"]["path"])  # type: ignore[index]
    expected = [out_dir / "train.jsonl", out_dir / "val.jsonl", out_dir / "test.jsonl"]
    if all(path.exists() and path.stat().st_size > 0 for path in expected):
        print(f"autotune: using existing data in {out_dir}", flush=True)
        return
    cmd = [args.python, "scripts/generate_dataset.py", "--config", args.data_config, "--render-preview"]
    print(f"autotune: generating missing data with {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def write_attempt_configs(
    config_root: Path,
    output_root: Path,
    attempt_name: str,
    base_train: Dict[str, Any],
    base_model: Dict[str, Any],
    recipe: Recipe,
) -> tuple[Path, Path, Path, Path]:
    train_cfg = json.loads(json.dumps(base_train))
    model_cfg = json.loads(json.dumps(base_model))
    train_cfg["lr"] = recipe.lr
    train_cfg.setdefault("loss", {})["edge_bce_weight"] = recipe.edge_bce_weight
    train_cfg.setdefault("loss", {})["node_bce_weight"] = recipe.node_bce_weight
    train_cfg.setdefault("checkpoint", {})["dir"] = str(output_root / attempt_name / "checkpoints")
    train_cfg.setdefault("logging", {})["wandb_run_name"] = f"surrogate-lattice24-balanced-{attempt_name}"
    model_cfg["dropout"] = recipe.dropout

    train_config_path = config_root / f"{attempt_name}_train.yaml"
    model_config_path = config_root / f"{attempt_name}_model.yaml"
    _write_yaml(train_config_path, train_cfg)
    _write_yaml(model_config_path, model_cfg)
    checkpoint_path = output_root / attempt_name / "checkpoints" / "surrogate_best.pt"
    eval_output = output_root / attempt_name / "rollouts"
    return train_config_path, model_config_path, checkpoint_path, eval_output


def run_training_attempt(
    args: argparse.Namespace,
    attempt_name: str,
    train_config_path: Path,
    model_config_path: Path,
    output_root: Path,
) -> Dict[str, Any]:
    log_path = output_root / f"{attempt_name}.log"
    monitor = Monitor(args.warmup_epochs, args.patience, args.min_delta, args.min_edge_f1)
    wandb_path: Optional[str] = None
    cmd = [
        args.python,
        "scripts/train_surrogate.py",
        "--data-config",
        args.data_config,
        "--model-config",
        _rel(model_config_path),
        "--train-config",
        _rel(train_config_path),
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    started_at = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    status = "running"
    decision = Decision(False, "continue", "training started")
    latest_metrics: Optional[Dict[str, float]] = None

    try:
        with log_path.open("w", encoding="utf-8") as log:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if wandb_path is None:
                    wandb_path = parse_wandb_path(line)
                parsed = parse_metric_line(line)
                if parsed is None:
                    continue
                latest_metrics = read_wandb_latest(wandb_path) if args.poll_wandb and wandb_path else None
                if latest_metrics is None:
                    latest_metrics = parsed
                decision = monitor.update(latest_metrics)
                if decision.stop:
                    status = "stopped_for_retune"
                    print(f"autotune: stopping {attempt_name}: {decision.reason} - {decision.detail}", flush=True)
                    terminate_process(proc)
                    break
    finally:
        if status == "running":
            return_code = proc.wait()
        else:
            return_code = proc.wait()

    if status == "running":
        status = "finished" if return_code == 0 else "failed"
    return {
        "attempt": attempt_name,
        "status": status,
        "return_code": return_code,
        "reason": decision.reason,
        "detail": decision.detail,
        "wandb_path": wandb_path,
        "elapsed_sec": round(time.time() - started_at, 1),
        "latest_metrics": latest_metrics,
        "log_path": str(log_path),
    }


def run_eval(args: argparse.Namespace, checkpoint_path: Path, eval_output: Path) -> None:
    cmd = [
        args.python,
        "scripts/eval_rollout.py",
        "--experiment",
        args.experiment,
        "--checkpoint",
        _rel(checkpoint_path),
        "--output-dir",
        _rel(eval_output),
    ]
    print(f"autotune: running evaluation with {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def propose_next_recipe(recipe: Recipe, result: Dict[str, Any]) -> Recipe:
    metrics = result.get("latest_metrics") or {}
    reason = str(result.get("reason", ""))
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))
    next_recipe = Recipe(**recipe.as_dict())

    if reason == "precision_low" or (recall > 0.90 and precision < 0.72):
        next_recipe.edge_bce_weight = max(0.45, recipe.edge_bce_weight * 0.75)
        next_recipe.lr = max(1e-4, recipe.lr * 0.85)
    elif reason == "recall_low" or (precision > 0.88 and recall < 0.70):
        next_recipe.edge_bce_weight = min(2.0, recipe.edge_bce_weight * 1.25)
        next_recipe.lr = max(1e-4, recipe.lr * 0.90)
    elif reason == "plateau":
        next_recipe.lr = max(1e-4, recipe.lr * 0.70)
        next_recipe.dropout = min(0.10, recipe.dropout + 0.02)
    else:
        next_recipe.lr = max(1e-4, recipe.lr * 0.80)
        next_recipe.dropout = min(0.10, recipe.dropout + 0.01)
    return next_recipe


def parse_metric_line(line: str) -> Optional[Dict[str, float]]:
    match = METRIC_RE.search(line)
    if not match:
        return None
    out = {key: float(value) for key, value in match.groupdict().items()}
    out["epoch"] = int(out["epoch"])
    out["selection_score"] = _selection_score(out)
    return out


def parse_wandb_path(line: str) -> Optional[str]:
    match = WANDB_RUN_RE.search(line)
    if not match:
        return None
    return f"{match.group('entity')}/{match.group('project')}/{match.group('run_id')}"


def read_wandb_latest(wandb_path: Optional[str]) -> Optional[Dict[str, float]]:
    if not wandb_path:
        return None
    try:
        import wandb

        api = wandb.Api()
        run = api.run(wandb_path)
        keys = [
            "epoch",
            "train/loss",
            "val/edge_bce",
            "val/node_bce",
            "val/edge_f1",
            "val/node_iou",
            "val/precision",
            "val/recall",
            "val/edge_prob_threshold",
            "selection_score",
        ]
        latest = None
        for row in run.scan_history(keys=keys):
            if row.get("val/edge_f1") is not None:
                latest = row
        if latest is None:
            return None
        return {
            "epoch": int(latest["epoch"]),
            "train_loss": float(latest.get("train/loss", 0.0)),
            "edge_bce": float(latest.get("val/edge_bce", 0.0)),
            "node_bce": float(latest.get("val/node_bce", 0.0)),
            "edge_f1": float(latest["val/edge_f1"]),
            "node_iou": float(latest.get("val/node_iou", 0.0)),
            "precision": float(latest.get("val/precision", 0.0)),
            "recall": float(latest.get("val/recall", 0.0)),
            "edge_prob_threshold": float(latest.get("val/edge_prob_threshold", 0.5)),
            "selection_score": float(latest.get("selection_score", 0.0)),
        }
    except Exception as exc:
        print(f"autotune: W&B read failed, using local metrics fallback: {exc}", flush=True)
        return None


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        proc.terminate()
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        proc.kill()


def _selection_score(metrics: Dict[str, float]) -> float:
    return 0.65 * float(metrics.get("edge_f1", 0.0)) + 0.35 * float(metrics.get("node_iou", 0.0))


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("Autotune config writing needs PyYAML. Install dependencies from requirements.txt.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


if __name__ == "__main__":
    main()
