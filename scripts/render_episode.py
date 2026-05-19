#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lattice_sim import ImpactAction, LatticeState
from src.viz import render_episode_svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a saved rollout JSON as SVG.")
    parser.add_argument("rollout_json")
    parser.add_argument("--output")
    parser.add_argument("--connectivity", default="eight_connected")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.rollout_json)
    data = json.loads(path.read_text(encoding="utf-8"))
    states = [LatticeState.from_jsonable(item, args.connectivity) for item in data["states"]]
    actions = [ImpactAction(**item) for item in data["actions"]]
    out = Path(args.output) if args.output else path.with_suffix(".svg")
    render_episode_svg(out, states, data["target"], actions, f"{data['method']} rollout")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
