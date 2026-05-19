from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


GraphRecord = Dict[str, object]


def write_jsonl(path: str | Path, records: Iterable[GraphRecord]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> List[GraphRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[GraphRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def split_records(records: Sequence[GraphRecord], num_train: int, num_val: int, num_test: int) -> Dict[str, List[GraphRecord]]:
    total = num_train + num_val + num_test
    if len(records) < total:
        raise ValueError(f"Need {total} records, got {len(records)}")
    return {
        "train": list(records[:num_train]),
        "val": list(records[num_train : num_train + num_val]),
        "test": list(records[num_train + num_val : total]),
    }


def shuffled_batches(records: Sequence[GraphRecord], batch_size: int, rng: random.Random) -> Iterator[List[GraphRecord]]:
    order = list(range(len(records)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [records[i] for i in order[start : start + batch_size]]


def has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


def collate_records(records: Sequence[GraphRecord], device: Optional[str] = None) -> Dict[str, object]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("Training needs PyTorch. Install dependencies from requirements.txt first.") from exc

    xs = []
    edge_attrs = []
    edge_indices = []
    us = []
    y_edges = []
    y_nodes = []
    node_batch = []
    edge_batch = []
    node_offset = 0

    for graph_id, record in enumerate(records):
        x = record["x"]  # type: ignore[index]
        edge_index = record["edge_index"]  # type: ignore[index]
        edge_attr = record["edge_attr"]  # type: ignore[index]
        n_nodes = len(x)  # type: ignore[arg-type]
        n_edges = len(edge_index)  # type: ignore[arg-type]
        xs.extend(x)  # type: ignore[arg-type]
        edge_attrs.extend(edge_attr)  # type: ignore[arg-type]
        edge_indices.extend([[src + node_offset, dst + node_offset] for src, dst in edge_index])  # type: ignore[union-attr]
        us.append(record["u"])  # type: ignore[arg-type]
        y_edges.extend(record["y_edge"])  # type: ignore[arg-type]
        y_nodes.extend(record["y_node"])  # type: ignore[arg-type]
        node_batch.extend([graph_id] * n_nodes)
        edge_batch.extend([graph_id] * n_edges)
        node_offset += n_nodes

    batch = {
        "x": torch.tensor(xs, dtype=torch.float32, device=device),
        "edge_index": torch.tensor(edge_indices, dtype=torch.long, device=device).t().contiguous(),
        "edge_attr": torch.tensor(edge_attrs, dtype=torch.float32, device=device),
        "u": torch.tensor(us, dtype=torch.float32, device=device),
        "y_edge": torch.tensor(y_edges, dtype=torch.float32, device=device),
        "y_node": torch.tensor(y_nodes, dtype=torch.float32, device=device),
        "node_batch": torch.tensor(node_batch, dtype=torch.long, device=device),
        "edge_batch": torch.tensor(edge_batch, dtype=torch.long, device=device),
    }
    return batch


def save_torch_dataset(path: str | Path, records: Sequence[GraphRecord]) -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(list(records), path)


def load_records(path: str | Path) -> List[GraphRecord]:
    path = Path(path)
    if path.suffix == ".pt":
        try:
            import torch

            return torch.load(path, map_location="cpu")
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyTorch is needed to load .pt datasets. Use the .jsonl split instead.") from exc
    return read_jsonl(path)
