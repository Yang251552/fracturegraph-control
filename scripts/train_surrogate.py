#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.graph_dataset import collate_records
from src.models import build_model
from src.metrics import shape_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MeshGraphNet-style fracture surrogate.")
    parser.add_argument("--data-config", default="configs/data/lattice_32.yaml")
    parser.add_argument("--model-config", default="configs/model/meshgnn_small.yaml")
    parser.add_argument("--train-config", default="configs/train/surrogate.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from torch.nn import functional as F
    except ModuleNotFoundError as exc:
        raise SystemExit("Training needs PyTorch. Install dependencies with `python3 -m pip install -r requirements.txt`.") from exc

    data_config = load_config(ROOT / args.data_config)
    model_config = load_config(ROOT / args.model_config)
    train_config = load_config(ROOT / args.train_config)
    data_dir = ROOT / (args.data_dir or str(data_config["output"]["path"]))  # type: ignore[index]
    train_records = JsonlRecordStore(data_dir / "train.jsonl", "train")
    val_records = JsonlRecordStore(data_dir / "val.jsonl", "val")

    seed = int(train_config.get("seed", 0))
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = _resolve_device(args.device, torch)
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("lr", 5e-4)),
        weight_decay=float(train_config.get("weight_decay", 1e-5)),
    )
    batch_size = int(train_config.get("batch_size", 16))
    epochs = int(train_config.get("epochs", 35))
    clip = float(train_config.get("gradient_clip", 1.0))
    loss_cfg = train_config.get("loss", {})
    edge_weight = float(loss_cfg.get("edge_bce_weight", 1.0))  # type: ignore[union-attr]
    node_weight = float(loss_cfg.get("node_bce_weight", 0.35))  # type: ignore[union-attr]
    logging_cfg = train_config.get("logging", {})
    print_every = int(logging_cfg.get("print_every_batches", 0)) if isinstance(logging_cfg, dict) else 0
    num_batches = max(1, math.ceil(len(train_records) / batch_size))
    ckpt_dir = ROOT / str(train_config["checkpoint"]["dir"])  # type: ignore[index]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb(args, data_config, model_config, train_config, data_dir)
    edge_pos_weight, node_pos_weight = _positive_weights(train_records, torch, device)
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "train_records": len(train_records),
                "val_records": len(val_records),
                "edge_pos_weight": float(edge_pos_weight.detach().cpu()),
                "node_pos_weight": float(node_pos_weight.detach().cpu()),
            }
        )

    best_f1 = -1.0
    history: List[Dict[str, float]] = []
    global_batch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses: List[float] = []
        for batch_idx, batch_records in enumerate(train_records.shuffled_batches(batch_size, rng), start=1):
            global_batch += 1
            batch = collate_records(batch_records, device=device)
            outputs = model(batch)
            edge_loss = F.binary_cross_entropy_with_logits(
                outputs["edge_logits"],
                batch["y_edge"],
                pos_weight=edge_pos_weight,
            )
            node_loss = F.binary_cross_entropy_with_logits(
                outputs["node_logits"],
                batch["y_node"],
                pos_weight=node_pos_weight,
            )
            loss = edge_weight * edge_loss + node_weight * node_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if print_every > 0 and (batch_idx % print_every == 0 or batch_idx == num_batches):
                recent = losses[-min(print_every, len(losses)) :]
                recent_loss = sum(recent) / max(1, len(recent))
                print(
                    f"epoch={epoch:03d} batch={batch_idx:04d}/{num_batches:04d} "
                    f"recent_train_loss={recent_loss:.4f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "epoch": epoch,
                            "batch": global_batch,
                            "train/batch_loss": recent_loss,
                            "train/epoch_progress": batch_idx / num_batches,
                            "lr": float(optimizer.param_groups[0]["lr"]),
                        },
                        step=global_batch,
                    )
        val_metrics = evaluate(model, val_records, batch_size, device)
        train_loss = sum(losses) / max(1, len(losses))
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"edge_bce={val_metrics['edge_bce']:.4f} node_bce={val_metrics['node_bce']:.4f} "
            f"edge_f1={val_metrics['edge_f1']:.3f} node_iou={val_metrics['node_iou']:.3f} "
            f"precision={val_metrics['precision']:.3f} recall={val_metrics['recall']:.3f} "
            f"edge_threshold={val_metrics['edge_prob_threshold']:.2f}"
        )
        selection = 0.65 * val_metrics["edge_f1"] + 0.35 * val_metrics["node_iou"]
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/edge_bce": val_metrics["edge_bce"],
                    "val/node_bce": val_metrics["node_bce"],
                    "val/edge_f1": val_metrics["edge_f1"],
                    "val/node_iou": val_metrics["node_iou"],
                    "val/precision": val_metrics["precision"],
                    "val/recall": val_metrics["recall"],
                    "val/edge_accuracy": val_metrics["edge_accuracy"],
                    "val/node_accuracy": val_metrics["node_accuracy"],
                    "val/edge_prob_threshold": val_metrics["edge_prob_threshold"],
                    "val/node_prob_threshold": val_metrics["node_prob_threshold"],
                    "selection_score": selection,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                },
                step=global_batch,
            )
        if selection > best_f1:
            best_f1 = selection
            save_checkpoint(ckpt_dir / "surrogate_best.pt", model, model_config, data_config, train_config, row, torch)
        if epoch % int(train_config.get("logging", {}).get("save_every", 5)) == 0:  # type: ignore[union-attr]
            save_checkpoint(ckpt_dir / f"surrogate_epoch_{epoch:03d}.pt", model, model_config, data_config, train_config, row, torch)

    save_checkpoint(ckpt_dir / "surrogate_last.pt", model, model_config, data_config, train_config, history[-1], torch)
    history_path = ckpt_dir / "train_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.summary.update(history[-1])
        wandb_run.save(str(history_path))
        wandb_run.save(str(ckpt_dir / "surrogate_best.pt"))
        wandb_run.finish()


class JsonlRecordStore:
    def __init__(self, path: Path, split_name: str):
        self.path = Path(path)
        self.split_name = split_name
        self.offsets = self._index_offsets()
        print(f"{self.split_name}: indexed {len(self.offsets)} records from {self.path}", flush=True)

    def __len__(self) -> int:
        return len(self.offsets)

    def iter_records(self) -> Iterator[Dict[str, object]]:
        with self.path.open("rb") as handle:
            for offset in self.offsets:
                handle.seek(offset)
                yield json.loads(handle.readline())

    def iter_batches(self, batch_size: int) -> Iterator[List[Dict[str, object]]]:
        batch: List[Dict[str, object]] = []
        for record in self.iter_records():
            batch.append(record)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def shuffled_batches(self, batch_size: int, rng: random.Random) -> Iterator[List[Dict[str, object]]]:
        order = list(range(len(self.offsets)))
        rng.shuffle(order)
        with self.path.open("rb") as handle:
            for start in range(0, len(order), batch_size):
                batch = []
                for record_idx in order[start : start + batch_size]:
                    handle.seek(self.offsets[record_idx])
                    batch.append(json.loads(handle.readline()))
                yield batch

    def _index_offsets(self) -> List[int]:
        offsets: List[int] = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
                    if len(offsets) % 10000 == 0:
                        print(f"{self.split_name}: indexed {len(offsets)} records", flush=True)
        return offsets


def _init_wandb(
    args: argparse.Namespace,
    data_config: Dict[str, Any],
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    data_dir: Path,
) -> Optional[Any]:
    logging_cfg = train_config.get("logging", {})
    if not isinstance(logging_cfg, dict) or not bool(logging_cfg.get("wandb", False)):
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise SystemExit("W&B logging is enabled. Install dependencies with `python3 -m pip install -r requirements.txt`.") from exc

    project = str(logging_cfg.get("wandb_project", "fracturegraph-control"))
    run_name = logging_cfg.get("wandb_run_name")
    run = wandb.init(
        project=project,
        name=str(run_name) if run_name else None,
        config={
            "data_config_path": args.data_config,
            "model_config_path": args.model_config,
            "train_config_path": args.train_config,
            "data_dir": str(data_dir),
            "data": data_config,
            "model": model_config,
            "train": train_config,
        },
    )
    run.define_metric("epoch")
    run.define_metric("batch")
    run.define_metric("train/*", step_metric="batch")
    run.define_metric("val/*", step_metric="batch")
    run.define_metric("selection_score", step_metric="batch")
    run.define_metric("lr", step_metric="batch")
    return run


def evaluate(model, records: JsonlRecordStore, batch_size: int, device: str) -> Dict[str, float]:
    import torch
    from torch.nn import functional as F

    model.eval()
    edge_losses: List[float] = []
    node_losses: List[float] = []
    edge_logits_all: List[float] = []
    edge_labels_all: List[float] = []
    node_logits_all: List[float] = []
    node_labels_all: List[float] = []
    node_batch_all: List[int] = []
    graph_offset = 0
    with torch.no_grad():
        for batch_records in records.iter_batches(batch_size):
            batch = collate_records(batch_records, device=device)
            outputs = model(batch)
            edge_loss = F.binary_cross_entropy_with_logits(outputs["edge_logits"], batch["y_edge"])
            node_loss = F.binary_cross_entropy_with_logits(outputs["node_logits"], batch["y_node"])
            edge_losses.append(float(edge_loss.detach().cpu()))
            node_losses.append(float(node_loss.detach().cpu()))
            edge_logits_all.extend(outputs["edge_logits"].detach().cpu().tolist())
            edge_labels_all.extend(batch["y_edge"].detach().cpu().tolist())
            node_logits_all.extend(outputs["node_logits"].detach().cpu().tolist())
            node_labels_all.extend(batch["y_node"].detach().cpu().tolist())
            node_batch_all.extend((batch["node_batch"].detach().cpu() + graph_offset).tolist())
            graph_offset += len(batch_records)
    edge_metrics = _edge_metrics(edge_logits_all, edge_labels_all)
    node_metrics = _node_metrics(node_logits_all, node_labels_all, node_batch_all)
    return {
        "edge_bce": sum(edge_losses) / max(1, len(edge_losses)),
        "node_bce": sum(node_losses) / max(1, len(node_losses)),
        **edge_metrics,
        **node_metrics,
    }


def save_checkpoint(path: Path, model, model_config, data_config, train_config, metrics, torch_module) -> None:
    torch_module.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config,
            "data_config": data_config,
            "train_config": train_config,
            "metrics": metrics,
        },
        path,
    )


def _positive_weights(records: JsonlRecordStore, torch_module, device: str):
    edge_positives = 0.0
    edge_total = 0.0
    node_positives = 0.0
    node_total = 0.0
    for idx, record in enumerate(records.iter_records(), start=1):
        edge_labels = record["y_edge"]
        node_labels = record["y_node"]
        edge_positives += sum(float(v) for v in edge_labels)  # type: ignore[union-attr]
        edge_total += len(edge_labels)  # type: ignore[arg-type]
        node_positives += sum(float(v) for v in node_labels)  # type: ignore[union-attr]
        node_total += len(node_labels)  # type: ignore[arg-type]
        if idx % 10000 == 0:
            print(f"train: scanned {idx}/{len(records)} records for class weights", flush=True)
    edge_weight = _positive_weight_value(edge_positives, edge_total)
    node_weight = _positive_weight_value(node_positives, node_total, clamp=(0.25, 4.0))
    print(f"class weights: edge_pos_weight={edge_weight:.4f} node_pos_weight={node_weight:.4f}", flush=True)
    return (
        torch_module.tensor(edge_weight, dtype=torch_module.float32, device=device),
        torch_module.tensor(node_weight, dtype=torch_module.float32, device=device),
    )


def _positive_weight_value(positives: float, total: float, clamp=None) -> float:
    negatives = max(1.0, total - positives)
    positives = max(1.0, positives)
    value = negatives / positives
    if clamp is not None:
        value = max(float(clamp[0]), min(float(clamp[1]), value))
    return value


def _edge_metrics(logits: List[float], labels: List[float]) -> Dict[str, float]:
    best = {"precision": 0.0, "recall": 0.0, "edge_f1": -1.0, "edge_accuracy": 0.0, "edge_prob_threshold": 0.5}
    for prob_threshold in [i / 100.0 for i in range(5, 96, 5)]:
        logit_threshold = _logit(prob_threshold)
        metrics = _edge_metrics_at_threshold(logits, labels, logit_threshold)
        if metrics["edge_f1"] > best["edge_f1"]:
            best = {**metrics, "edge_prob_threshold": prob_threshold}
    default = _edge_metrics_at_threshold(logits, labels, 0.0)
    return {
        **best,
        "edge_f1_at_05": default["edge_f1"],
        "precision_at_05": default["precision"],
        "recall_at_05": default["recall"],
    }


def _edge_metrics_at_threshold(logits: List[float], labels: List[float], threshold: float) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for logit, label in zip(logits, labels):
        pred = logit >= threshold
        truth = label >= 0.5
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    edge_f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {"precision": precision, "recall": recall, "edge_f1": edge_f1, "edge_accuracy": accuracy}


def _node_metrics(logits: List[float], labels: List[float], node_batch: List[int]) -> Dict[str, float]:
    best = {"node_iou": -1.0, "node_accuracy": 0.0, "node_prob_threshold": 0.5}
    for prob_threshold in [i / 100.0 for i in range(25, 96, 5)]:
        preds = [_sigmoid(logit) >= prob_threshold for logit in logits]
        truths = [label >= 0.5 for label in labels]
        iou = _mean_graph_iou(preds, truths, node_batch)
        accuracy = sum(int(pred == truth) for pred, truth in zip(preds, truths)) / max(1, len(truths))
        if iou > best["node_iou"]:
            best = {"node_iou": iou, "node_accuracy": accuracy, "node_prob_threshold": prob_threshold}
    return best


def _mean_graph_iou(preds: List[bool], truths: List[bool], node_batch: List[int]) -> float:
    by_graph: Dict[int, Dict[str, List[bool]]] = {}
    for pred, truth, graph_id in zip(preds, truths, node_batch):
        bucket = by_graph.setdefault(int(graph_id), {"pred": [], "truth": []})
        bucket["pred"].append(pred)
        bucket["truth"].append(truth)
    values = [shape_iou(bucket["pred"], bucket["truth"]) for bucket in by_graph.values()]
    return sum(values) / max(1, len(values))


def _sigmoid(value: float) -> float:
    import math

    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _logit(probability: float) -> float:
    import math

    p = min(0.999, max(0.001, probability))
    return math.log(p / (1.0 - p))


def _resolve_device(requested: str, torch_module) -> str:
    if requested != "auto":
        return requested
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


if __name__ == "__main__":
    main()
