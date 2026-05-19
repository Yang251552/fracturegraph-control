from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .graph_dataset import collate_records
from .lattice_sim import ImpactAction, LatticeState, sample_action, step, update_connected_component
from .metrics import planner_score, shape_iou


RolloutFn = Callable[[LatticeState, Sequence[ImpactAction]], LatticeState]


@dataclass
class PlanResult:
    method: str
    actions: List[ImpactAction]
    score: float
    final_iou: float
    states: List[LatticeState]


def rollout_simulator(
    initial: LatticeState,
    actions: Sequence[ImpactAction],
    sim_config: Dict[str, object],
    rng: random.Random,
) -> Tuple[LatticeState, List[LatticeState]]:
    states = [initial]
    current = initial
    for action in actions:
        current = step(current, action, sim_config, rng)
        states.append(current)
    return current, states


class SurrogateRollout:
    def __init__(
        self,
        model,
        threshold: float = 0.45,
        node_threshold: float = 0.5,
        use_node_head: bool = True,
        device: str = "cpu",
    ):
        self.model = model
        self.threshold = threshold
        self.node_threshold = node_threshold
        self.use_node_head = use_node_head
        self.device = device
        self.model.eval()

    def __call__(self, initial: LatticeState, actions: Sequence[ImpactAction]) -> Tuple[LatticeState, List[LatticeState]]:
        import torch

        states = [initial]
        current = initial
        with torch.no_grad():
            for action in actions:
                batch = collate_records([current.to_record(action)], device=self.device)
                outputs = self.model(batch)
                if isinstance(outputs, dict):
                    edge_logits = outputs["edge_logits"]
                    node_logits = outputs.get("node_logits")
                else:
                    edge_logits = outputs
                    node_logits = None
                probs = torch.sigmoid(edge_logits).detach().cpu().tolist()
                breaks = [
                    (prob >= self.threshold) and (not current.edge_broken[i])
                    for i, prob in enumerate(probs)
                ]
                current = current.apply_predicted_breaks(breaks)
                if self.use_node_head and node_logits is not None:
                    node_probs = torch.sigmoid(node_logits).detach().cpu().tolist()
                    for node_id, prob in enumerate(node_probs):
                        if current.alive[node_id] and prob < self.node_threshold:
                            current.alive[node_id] = False
                    update_connected_component(current)
                states.append(current)
        return current, states


def random_plan(
    initial: LatticeState,
    target: Sequence[int],
    sim_config: Dict[str, object],
    planner_config: Dict[str, object],
    rng: random.Random,
) -> PlanResult:
    horizon = int(planner_config.get("horizon", 5))
    actions = [sample_action(sim_config, rng) for _ in range(horizon)]
    final, states = rollout_simulator(initial, actions, sim_config, rng)
    objective = planner_config.get("objective", {})  # type: ignore[assignment]
    score = planner_score(final.alive_mask(), target, actions, objective)  # type: ignore[arg-type]
    return PlanResult("random", actions, score, shape_iou(final.alive_mask(), target), states)


def greedy_plan(
    initial: LatticeState,
    target: Sequence[int],
    sim_config: Dict[str, object],
    planner_config: Dict[str, object],
    rng: random.Random,
    rollout: Optional[Callable[[LatticeState, Sequence[ImpactAction]], Tuple[LatticeState, List[LatticeState]]]] = None,
    candidates: int = 72,
) -> PlanResult:
    horizon = int(planner_config.get("horizon", 5))
    objective = planner_config.get("objective", {})  # type: ignore[assignment]
    actions: List[ImpactAction] = []
    current = initial
    states = [initial]
    rollout_fn = rollout or (lambda s, a: rollout_simulator(s, a, sim_config, rng))
    for _ in range(horizon):
        best_action = None
        best_score = -1e9
        best_state = current
        for _ in range(candidates):
            candidate = sample_action(sim_config, rng)
            final, _ = rollout_fn(current, [candidate])
            score = planner_score(final.alive_mask(), target, actions + [candidate], objective)  # type: ignore[arg-type]
            if score > best_score:
                best_score = score
                best_action = candidate
                best_state = final
        if best_action is None:
            break
        actions.append(best_action)
        current = best_state
        states.append(current)
    final_true, true_states = rollout_simulator(initial, actions, sim_config, rng)
    score = planner_score(final_true.alive_mask(), target, actions, objective)  # type: ignore[arg-type]
    return PlanResult("greedy", actions, score, shape_iou(final_true.alive_mask(), target), true_states)


def cem_plan(
    initial: LatticeState,
    target: Sequence[int],
    sim_config: Dict[str, object],
    planner_config: Dict[str, object],
    rng: random.Random,
    rollout: Callable[[LatticeState, Sequence[ImpactAction]], Tuple[LatticeState, List[LatticeState]]],
    method: str = "cem_surrogate",
) -> PlanResult:
    horizon = int(planner_config.get("horizon", 5))
    population = int(planner_config.get("population", 160))
    iterations = int(planner_config.get("iterations", 5))
    elite_frac = float(planner_config.get("elite_frac", 0.12))
    n_elite = max(2, int(population * elite_frac))
    objective = planner_config.get("objective", {})  # type: ignore[assignment]
    action_space = planner_config.get("action_space", {})  # type: ignore[assignment]
    x_low, x_high = [float(v) for v in action_space.get("impact_x", [0, initial.grid_size - 1])]  # type: ignore[union-attr]
    y_low, y_high = [float(v) for v in action_space.get("impact_y", [0, initial.grid_size - 1])]  # type: ignore[union-attr]
    f_low, f_high = [float(v) for v in action_space.get("force", [0.2, 1.0])]  # type: ignore[union-attr]

    means = [[0.5 * (x_low + x_high), 0.5 * (y_low + y_high), 0.75 * f_high] for _ in range(horizon)]
    stds = [[0.35 * (x_high - x_low), 0.35 * (y_high - y_low), 0.35 * (f_high - f_low)] for _ in range(horizon)]
    best_actions: List[ImpactAction] = []
    best_score = -1e9

    for _ in range(iterations):
        scored: List[Tuple[float, List[ImpactAction]]] = []
        for _ in range(population):
            actions = [
                ImpactAction(
                    impact_x=_clip(rng.gauss(means[t][0], stds[t][0]), x_low, x_high),
                    impact_y=_clip(rng.gauss(means[t][1], stds[t][1]), y_low, y_high),
                    force=_clip(rng.gauss(means[t][2], stds[t][2]), f_low, f_high),
                )
                for t in range(horizon)
            ]
            final, _ = rollout(initial, actions)
            score = planner_score(final.alive_mask(), target, actions, objective)  # type: ignore[arg-type]
            scored.append((score, actions))
            if score > best_score:
                best_score = score
                best_actions = actions
        scored.sort(key=lambda item: item[0], reverse=True)
        elites = [actions for _, actions in scored[:n_elite]]
        for t in range(horizon):
            for dim in range(3):
                values = [_action_vector(actions[t])[dim] for actions in elites]
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                means[t][dim] = mean
                stds[t][dim] = max(math.sqrt(variance), [0.35, 0.35, 0.03][dim])

    final_true, true_states = rollout_simulator(initial, best_actions, sim_config, rng)
    true_score = planner_score(final_true.alive_mask(), target, best_actions, objective)  # type: ignore[arg-type]
    return PlanResult(method, best_actions, true_score, shape_iou(final_true.alive_mask(), target), true_states)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _action_vector(action: ImpactAction) -> List[float]:
    return [action.impact_x, action.impact_y, action.force]
