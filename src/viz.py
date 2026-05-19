from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .lattice_sim import ImpactAction, LatticeState
from .metrics import shape_iou


PALETTE = {
    "rock": "#60646c",
    "rock_edge": "#30333a",
    "target_empty": "#eef1f4",
    "unwanted": "#d89b45",
    "overbreak": "#bf5f5f",
    "grid": "#ffffff",
    "impact": "#2b6cb0",
    "text": "#222831",
}


def render_episode_svg(
    path: str | Path,
    states: Sequence[LatticeState],
    target: Optional[Sequence[int]] = None,
    actions: Optional[Sequence[ImpactAction]] = None,
    title: str = "Fracture rollout",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("initial", states[0]),
        ("target", None),
        ("final", states[-1]),
    ]
    if len(states) > 2:
        mid = states[min(len(states) - 1, max(1, len(states) // 2))]
        panels.insert(2, ("mid rollout", mid))

    cell = 10
    gap = 28
    label_h = 26
    n = states[0].grid_size
    panel_w = n * cell
    panel_h = n * cell + label_h
    width = len(panels) * panel_w + (len(panels) + 1) * gap
    action_h = 62 if actions else 0
    height = panel_h + action_h + 72

    parts: List[str] = [
        _svg_header(width, height),
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="{gap}" y="34" font-size="20" font-family="Inter,Arial" font-weight="700" fill="{PALETTE["text"]}">{html.escape(title)}</text>',
    ]
    if target is not None:
        final_iou = shape_iou(states[-1].alive_mask(), target)
        parts.append(
            f'<text x="{gap}" y="56" font-size="12" font-family="Inter,Arial" fill="#5b6470">final IoU: {final_iou:.3f}</text>'
        )

    for panel_id, (label, state) in enumerate(panels):
        x0 = gap + panel_id * (panel_w + gap)
        y0 = 72
        if label == "target" and target is not None:
            parts.append(_render_mask(label, target, n, x0, y0, cell))
        elif state is not None:
            parts.append(_render_state(label, state, target, x0, y0, cell))

    if actions:
        y = 72 + panel_h + 30
        parts.append(f'<text x="{gap}" y="{y}" font-size="13" font-family="Inter,Arial" font-weight="700" fill="{PALETTE["text"]}">planned impacts</text>')
        for i, action in enumerate(actions):
            x = gap + 112 * i
            parts.append(
                f'<text x="{x}" y="{y + 23}" font-size="11" font-family="Inter,Arial" fill="#384252">'
                f'{i + 1}: x={action.impact_x:.1f}, y={action.impact_y:.1f}, f={action.force:.2f}</text>'
            )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_rollout_json(
    path: str | Path,
    method: str,
    states: Sequence[LatticeState],
    target: Sequence[int],
    actions: Sequence[ImpactAction],
    metrics: Dict[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "grid_size": states[0].grid_size,
        "target": list(target),
        "actions": [action.as_dict() for action in actions],
        "states": [state.to_jsonable() for state in states],
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_report_html(path: str | Path, rows: Sequence[Dict[str, object]], figure_paths: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in row.values())
            + "</tr>"
        )
    figures = "\n".join(
        f'<figure><img src="{html.escape(src)}" alt="rollout"><figcaption>{html.escape(Path(src).name)}</figcaption></figure>'
        for src in figure_paths
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FractureGraph-Control Report</title>
  <style>
    body {{ margin: 0; font-family: Inter, Arial, sans-serif; color: #20242a; background: #f7f9fb; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 36px 24px 60px; }}
    h1 {{ font-size: 30px; margin: 0 0 8px; }}
    p {{ color: #56606d; line-height: 1.55; }}
    table {{ border-collapse: collapse; background: white; width: 100%; margin: 24px 0; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e6e9ee; text-align: left; font-size: 14px; }}
    th {{ background: #eef2f6; }}
    figure {{ margin: 26px 0; background: white; padding: 16px; border: 1px solid #e3e7ed; }}
    img {{ max-width: 100%; display: block; }}
    figcaption {{ margin-top: 8px; color: #687381; font-size: 13px; }}
  </style>
</head>
<body>
<main>
  <h1>FractureGraph-Control</h1>
  <p>Compact 2D fracture-control prototype with graph data, learned surrogate hooks, CEM planning, and rollout diagnostics.</p>
  <table>
    <thead><tr>{''.join(f'<th>{html.escape(str(k))}</th>' for k in (rows[0].keys() if rows else []))}</tr></thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>
  {figures}
</main>
</body>
</html>"""
    path.write_text(html_text, encoding="utf-8")


def _svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def _render_state(label: str, state: LatticeState, target: Optional[Sequence[int]], x0: int, y0: int, cell: int) -> str:
    parts = [
        f'<text x="{x0}" y="{y0 - 9}" font-size="13" font-family="Inter,Arial" font-weight="700" fill="{PALETTE["text"]}">{html.escape(label)}</text>'
    ]
    n = state.grid_size
    for y in range(n):
        for x in range(n):
            idx = y * n + x
            alive = bool(state.alive[idx])
            desired = bool(target[idx]) if target is not None else alive
            color = _cell_color(alive, desired, target is not None)
            parts.append(
                f'<rect x="{x0 + x * cell}" y="{y0 + y * cell}" width="{cell - 1}" height="{cell - 1}" fill="{color}"/>'
            )
    return "\n".join(parts)


def _render_mask(label: str, mask: Sequence[int], n: int, x0: int, y0: int, cell: int) -> str:
    parts = [
        f'<text x="{x0}" y="{y0 - 9}" font-size="13" font-family="Inter,Arial" font-weight="700" fill="{PALETTE["text"]}">{html.escape(label)}</text>'
    ]
    for y in range(n):
        for x in range(n):
            idx = y * n + x
            color = PALETTE["rock"] if mask[idx] else PALETTE["target_empty"]
            parts.append(
                f'<rect x="{x0 + x * cell}" y="{y0 + y * cell}" width="{cell - 1}" height="{cell - 1}" fill="{color}"/>'
            )
    return "\n".join(parts)


def _cell_color(alive: bool, desired: bool, has_target: bool) -> str:
    if not has_target:
        return PALETTE["rock"] if alive else PALETTE["target_empty"]
    if alive and desired:
        return PALETTE["rock"]
    if alive and not desired:
        return PALETTE["unwanted"]
    if not alive and desired:
        return PALETTE["overbreak"]
    return PALETTE["target_empty"]
