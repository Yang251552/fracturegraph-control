from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ImpactAction:
    impact_x: float
    impact_y: float
    force: float

    def as_features(self, grid_size: int) -> List[float]:
        denom = max(1, grid_size - 1)
        return [self.impact_x / denom, self.impact_y / denom, self.force]

    def as_dict(self) -> Dict[str, float]:
        return {"impact_x": self.impact_x, "impact_y": self.impact_y, "force": self.force}


@dataclass(frozen=True)
class Edge:
    src: int
    dst: int
    dx: float
    dy: float
    length: float


class LatticeState:
    def __init__(
        self,
        grid_size: int,
        edges: Sequence[Edge],
        alive: Sequence[bool],
        node_damage: Sequence[float],
        edge_broken: Sequence[bool],
        edge_damage: Sequence[float],
        edge_threshold: Sequence[float],
    ):
        self.grid_size = grid_size
        self.edges = list(edges)
        self.alive = list(alive)
        self.node_damage = list(node_damage)
        self.edge_broken = list(edge_broken)
        self.edge_damage = list(edge_damage)
        self.edge_threshold = list(edge_threshold)

    def clone(self) -> "LatticeState":
        return LatticeState(
            self.grid_size,
            self.edges,
            self.alive,
            self.node_damage,
            self.edge_broken,
            self.edge_damage,
            self.edge_threshold,
        )

    @property
    def num_nodes(self) -> int:
        return self.grid_size * self.grid_size

    def node_xy(self, idx: int) -> Tuple[int, int]:
        return idx % self.grid_size, idx // self.grid_size

    def alive_mask(self) -> List[int]:
        return [1 if value else 0 for value in self.alive]

    def node_features(self) -> List[List[float]]:
        n = self.grid_size
        denom = max(1, n - 1)
        features: List[List[float]] = []
        for idx in range(self.num_nodes):
            x, y = self.node_xy(idx)
            boundary = 1.0 if x in {0, n - 1} or y in {0, n - 1} else 0.0
            features.append(
                [
                    x / denom,
                    y / denom,
                    boundary,
                    1.0 if self.alive[idx] else 0.0,
                    float(self.node_damage[idx]),
                ]
            )
        return features

    def edge_index(self) -> List[List[int]]:
        return [[edge.src, edge.dst] for edge in self.edges]

    def edge_features(self) -> List[List[float]]:
        denom = max(1, self.grid_size - 1)
        return [
            [
                edge.dx / denom,
                edge.dy / denom,
                edge.length / denom,
                1.0 if self.edge_broken[i] else 0.0,
            ]
            for i, edge in enumerate(self.edges)
        ]

    def to_record(self, action: ImpactAction, next_state: Optional["LatticeState"] = None) -> Dict[str, object]:
        y_edge = [0.0] * len(self.edges)
        y_node = [1.0 if value else 0.0 for value in self.alive]
        if next_state is not None:
            y_edge = [
                1.0 if (not before and after) else 0.0
                for before, after in zip(self.edge_broken, next_state.edge_broken)
            ]
            y_node = [1.0 if value else 0.0 for value in next_state.alive]
        return {
            "grid_size": self.grid_size,
            "x": self.node_features(),
            "edge_index": self.edge_index(),
            "edge_attr": self.edge_features(),
            "u": action.as_features(self.grid_size),
            "y_edge": y_edge,
            "y_node": y_node,
            "alive": self.alive_mask(),
            "next_alive": y_node,
        }

    def apply_predicted_breaks(self, break_mask: Sequence[bool]) -> "LatticeState":
        out = self.clone()
        for i, should_break in enumerate(break_mask):
            if should_break and not out.edge_broken[i]:
                out.edge_broken[i] = True
        update_connected_component(out)
        return out

    def to_jsonable(self) -> Dict[str, object]:
        return {
            "grid_size": self.grid_size,
            "alive": self.alive_mask(),
            "node_damage": self.node_damage,
            "edge_broken": [1 if value else 0 for value in self.edge_broken],
            "edge_damage": self.edge_damage,
            "edge_threshold": self.edge_threshold,
        }

    @classmethod
    def from_jsonable(cls, data: Dict[str, object], connectivity: str = "eight_connected") -> "LatticeState":
        grid_size = int(data["grid_size"])
        edges = build_edges(grid_size, connectivity)
        return cls(
            grid_size,
            edges,
            [bool(v) for v in data["alive"]],  # type: ignore[index]
            [float(v) for v in data["node_damage"]],  # type: ignore[index]
            [bool(v) for v in data["edge_broken"]],  # type: ignore[index]
            [float(v) for v in data["edge_damage"]],  # type: ignore[index]
            [float(v) for v in data["edge_threshold"]],  # type: ignore[index]
        )


def build_edges(grid_size: int, connectivity: str = "eight_connected") -> List[Edge]:
    edges: List[Edge] = []
    offsets = [(1, 0), (0, 1)]
    if connectivity == "eight_connected":
        offsets.extend([(1, 1), (1, -1)])
    for y in range(grid_size):
        for x in range(grid_size):
            src = y * grid_size + x
            for dx, dy in offsets:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    dst = ny * grid_size + nx
                    edges.append(Edge(src=src, dst=dst, dx=float(dx), dy=float(dy), length=math.hypot(dx, dy)))
    return edges


def create_initial_state(config: Dict[str, object], rng: random.Random) -> LatticeState:
    grid_size = int(config["grid_size"])
    connectivity = str(config.get("connectivity", "eight_connected"))
    edges = build_edges(grid_size, connectivity)
    material = config.get("material", {})  # type: ignore[assignment]
    mean = float(material.get("fracture_threshold_mean", 0.55))  # type: ignore[union-attr]
    std = float(material.get("fracture_threshold_std", 0.08))  # type: ignore[union-attr]
    max_damage = float(config.get("max_initial_damage", 0.05))
    thresholds = [max(0.05, rng.gauss(mean, std)) for _ in edges]
    edge_damage = [rng.random() * max_damage for _ in edges]
    node_damage = [rng.random() * max_damage for _ in range(grid_size * grid_size)]
    return LatticeState(
        grid_size=grid_size,
        edges=edges,
        alive=[True for _ in range(grid_size * grid_size)],
        node_damage=node_damage,
        edge_broken=[False for _ in edges],
        edge_damage=edge_damage,
        edge_threshold=thresholds,
    )


def sample_action(config: Dict[str, object], rng: random.Random) -> ImpactAction:
    grid_size = int(config["grid_size"])
    impact = config.get("impact", {})  # type: ignore[assignment]
    force_min = float(impact.get("force_min", 0.2))  # type: ignore[union-attr]
    force_max = float(impact.get("force_max", 1.0))  # type: ignore[union-attr]
    margin = max(1.0, grid_size * 0.04)
    return ImpactAction(
        impact_x=rng.uniform(margin, grid_size - 1 - margin),
        impact_y=rng.uniform(margin, grid_size - 1 - margin),
        force=rng.uniform(force_min, force_max),
    )


def step(state: LatticeState, action: ImpactAction, config: Dict[str, object], rng: random.Random) -> LatticeState:
    out = state.clone()
    impact_cfg = config.get("impact", {})  # type: ignore[assignment]
    material = config.get("material", {})  # type: ignore[assignment]
    radius = float(impact_cfg.get("radius", 3.0))  # type: ignore[union-attr]
    noise_std = float(material.get("noise_std", 0.03))  # type: ignore[union-attr]
    damage_decay = float(material.get("damage_decay", 0.85))  # type: ignore[union-attr]
    damage_gain = float(material.get("damage_gain", 0.45))  # type: ignore[union-attr]

    for i, edge in enumerate(out.edges):
        if out.edge_broken[i] or not (out.alive[edge.src] and out.alive[edge.dst]):
            continue
        sx, sy = out.node_xy(edge.src)
        dx, dy = out.node_xy(edge.dst)
        mid_x = 0.5 * (sx + dx)
        mid_y = 0.5 * (sy + dy)
        dist_sq = (mid_x - action.impact_x) ** 2 + (mid_y - action.impact_y) ** 2
        local = action.force * math.exp(-dist_sq / max(1e-6, radius * radius))
        angle = math.atan2(mid_y - action.impact_y, mid_x - action.impact_x + 1e-6)
        orientation = math.atan2(edge.dy, edge.dx + 1e-6)
        anisotropy = 1.0 + 0.18 * math.cos(2.0 * (angle - orientation))
        stress = max(0.0, local * anisotropy + rng.gauss(0.0, noise_std))
        out.edge_damage[i] = damage_decay * out.edge_damage[i] + damage_gain * stress
        if stress + out.edge_damage[i] > out.edge_threshold[i]:
            out.edge_broken[i] = True
        contribution = min(1.0, stress)
        out.node_damage[edge.src] = max(out.node_damage[edge.src], contribution)
        out.node_damage[edge.dst] = max(out.node_damage[edge.dst], contribution)

    update_connected_component(out)
    return out


def update_connected_component(state: LatticeState) -> None:
    alive_nodes = [idx for idx, alive in enumerate(state.alive) if alive]
    if not alive_nodes:
        return
    adjacency: List[List[int]] = [[] for _ in range(state.num_nodes)]
    for i, edge in enumerate(state.edges):
        if state.edge_broken[i]:
            continue
        if state.alive[edge.src] and state.alive[edge.dst]:
            adjacency[edge.src].append(edge.dst)
            adjacency[edge.dst].append(edge.src)

    visited = [False] * state.num_nodes
    components: List[List[int]] = []
    for start in alive_nodes:
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: List[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for nxt in adjacency[node]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        components.append(component)

    main = set(max(components, key=len))
    for idx in alive_nodes:
        if idx not in main:
            state.alive[idx] = False
    for i, edge in enumerate(state.edges):
        if not (state.alive[edge.src] and state.alive[edge.dst]):
            state.edge_broken[i] = True


def target_mask(grid_size: int, target_type: str, margin: int = 4) -> List[int]:
    keep: List[int] = []
    cx = (grid_size - 1) / 2.0
    cy = (grid_size - 1) / 2.0
    radius = max(2.0, grid_size * 0.22)
    for y in range(grid_size):
        for x in range(grid_size):
            desired = True
            if target_type == "triangle_wedge":
                desired = not (x + y > 2 * grid_size - margin - 4)
            elif target_type == "diagonal_bevel":
                desired = x - y < grid_size - margin
            elif target_type == "rectangle_cut":
                desired = not (x > grid_size - margin - 1 and y < grid_size // 2)
            elif target_type == "circular_notch":
                desired = (x - cx) ** 2 + (y - cy) ** 2 > radius * radius or y < cy
            else:
                raise ValueError(f"Unknown target type: {target_type}")
            keep.append(1 if desired else 0)
    return keep


def save_state(path: str | Path, state: LatticeState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_jsonable(), indent=2), encoding="utf-8")


def load_state(path: str | Path, connectivity: str = "eight_connected") -> LatticeState:
    return LatticeState.from_jsonable(json.loads(Path(path).read_text(encoding="utf-8")), connectivity)


def make_rollout(
    initial: LatticeState,
    actions: Iterable[ImpactAction],
    config: Dict[str, object],
    rng: random.Random,
) -> List[LatticeState]:
    states = [initial]
    current = initial
    for action in actions:
        current = step(current, action, config, rng)
        states.append(current)
    return states
