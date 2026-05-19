#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.lattice_sim import create_initial_state, sample_action, step, target_mask
from src.viz import render_episode_svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate graph-structured fracture transitions.")
    parser.add_argument("--config", default="configs/data/lattice_32.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-train", type=int)
    parser.add_argument("--num-val", type=int)
    parser.add_argument("--num-test", type=int)
    parser.add_argument("--output")
    parser.add_argument("--render-preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(ROOT / args.config)
    if args.num_train is not None:
        config["num_train"] = args.num_train
    if args.num_val is not None:
        config["num_val"] = args.num_val
    if args.num_test is not None:
        config["num_test"] = args.num_test
    if args.output:
        config.setdefault("output", {})["path"] = args.output

    rng = random.Random(args.seed)
    num_train = int(config["num_train"])
    num_val = int(config["num_val"])
    num_test = int(config["num_test"])
    total = num_train + num_val + num_test
    out_dir = ROOT / str(config["output"]["path"])  # type: ignore[index]
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [("train", num_train), ("val", num_val), ("test", num_test)]
    split_counts = write_splits(out_dir, config, splits, rng)

    metadata = {
        "seed": args.seed,
        "config": config,
        "splits": split_counts,
        "node_features": ["x", "y", "boundary", "alive", "damage"],
        "edge_features": ["dx", "dy", "length", "broken"],
        "global_features": ["impact_x", "impact_y", "impact_force"],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.render_preview:
        preview_rng = random.Random(args.seed + 17)
        state = create_initial_state(config, preview_rng)
        actions = [sample_action(config, preview_rng) for _ in range(int(config.get("episode_length", 5)))]
        states = [state]
        for action in actions:
            state = step(state, action, config, preview_rng)
            states.append(state)
        target = target_mask(int(config["grid_size"]), "triangle_wedge", margin=max(2, int(config["grid_size"]) // 8))
        render_episode_svg(ROOT / "results/figures/dataset_preview.svg", states, target, actions, "Synthetic fracture transitions")

    print(f"Wrote {total} samples to {out_dir}")


def write_splits(
    out_dir: Path,
    config: Dict[str, object],
    splits: List[Tuple[str, int]],
    rng: random.Random,
) -> Dict[str, int]:
    records = iter_records(config, sum(count for _, count in splits), rng)
    counts: Dict[str, int] = {}
    for split_name, split_count in splits:
        path = out_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for idx in range(split_count):
                record = next(records)
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                if (idx + 1) % 10000 == 0:
                    print(f"{split_name}: wrote {idx + 1}/{split_count} records", flush=True)
        counts[split_name] = split_count
        print(f"{split_name}: wrote {split_count} records to {path}", flush=True)
    return counts


def iter_records(config: Dict[str, object], total: int, rng: random.Random) -> Iterator[Dict[str, object]]:
    episode_len = int(config.get("episode_length", 5))
    state = create_initial_state(config, rng)
    step_id = 0

    for _ in range(total):
        action = sample_action(config, rng)
        next_state = step(state, action, config, rng)
        yield state.to_record(action, next_state)
        state = next_state
        step_id += 1
        alive_fraction = sum(state.alive) / max(1, state.num_nodes)
        if step_id >= episode_len or alive_fraction < 0.58:
            state = create_initial_state(config, rng)
            step_id = 0


def generate_records(config: Dict[str, object], total: int, rng: random.Random) -> List[Dict[str, object]]:
    return list(iter_records(config, total, rng))


if __name__ == "__main__":
    main()
