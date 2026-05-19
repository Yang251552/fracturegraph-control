from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from .lattice_sim import ImpactAction


def shape_iou(alive: Sequence[int] | Sequence[bool], target: Sequence[int] | Sequence[bool]) -> float:
    intersection = 0
    union = 0
    for a, t in zip(alive, target):
        av = bool(a)
        tv = bool(t)
        if av and tv:
            intersection += 1
        if av or tv:
            union += 1
    return intersection / union if union else 1.0


def shape_error(alive: Sequence[int] | Sequence[bool], target: Sequence[int] | Sequence[bool]) -> float:
    return 1.0 - shape_iou(alive, target)


def removed_desired_fraction(alive: Sequence[int] | Sequence[bool], target: Sequence[int] | Sequence[bool]) -> float:
    desired = 0
    removed = 0
    for a, t in zip(alive, target):
        if bool(t):
            desired += 1
            if not bool(a):
                removed += 1
    return removed / desired if desired else 0.0


def action_cost(actions: Iterable[ImpactAction]) -> float:
    values = [action.force for action in actions]
    return sum(values) / max(1, len(values))


def planner_score(
    alive: Sequence[int] | Sequence[bool],
    target: Sequence[int] | Sequence[bool],
    actions: Sequence[ImpactAction],
    objective: Dict[str, float],
) -> float:
    iou_weight = float(objective.get("target_iou_weight", 1.0))
    cost_weight = float(objective.get("action_cost_weight", 0.02))
    overbreak_weight = float(objective.get("overbreak_penalty", 0.5))
    return (
        iou_weight * shape_iou(alive, target)
        - cost_weight * action_cost(actions)
        - overbreak_weight * removed_desired_fraction(alive, target)
    )


def edge_classification_metrics(logits: List[float], labels: List[float], threshold: float = 0.0) -> Dict[str, float]:
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
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
