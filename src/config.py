from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config, with a tiny fallback parser for the project configs."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)


def save_json(path: str | Path, data: Any) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for i, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:i]
    return line


def _preprocess(text: str) -> List[Tuple[int, str]]:
    lines: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        clean = _strip_comment(raw).rstrip()
        if not clean.strip():
            continue
        indent = len(clean) - len(clean.lstrip(" "))
        lines.append((indent, clean.strip()))
    return lines


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    lines = _preprocess(text)
    if not lines:
        return {}
    value, next_i = _parse_block(lines, 0, lines[0][0])
    if next_i != len(lines):
        raise ValueError("Could not parse entire config.")
    if not isinstance(value, dict):
        raise ValueError("Top-level config must be a mapping.")
    return value


def _parse_block(lines: List[Tuple[int, str]], i: int, indent: int) -> Tuple[Any, int]:
    if i >= len(lines):
        return {}, i
    if lines[i][1].startswith("- "):
        return _parse_list(lines, i, indent)
    return _parse_dict(lines, i, indent)


def _parse_dict(lines: List[Tuple[int, str]], i: int, indent: int) -> Tuple[Dict[str, Any], int]:
    out: Dict[str, Any] = {}
    while i < len(lines):
        cur_indent, text = lines[i]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise ValueError(f"Unexpected indentation near: {text}")
        if ":" not in text:
            raise ValueError(f"Expected key/value near: {text}")
        key, raw_value = text.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            out[key] = _parse_scalar(raw_value)
            i += 1
        else:
            if i + 1 >= len(lines) or lines[i + 1][0] <= cur_indent:
                out[key] = {}
                i += 1
            else:
                out[key], i = _parse_block(lines, i + 1, lines[i + 1][0])
    return out, i


def _parse_list(lines: List[Tuple[int, str]], i: int, indent: int) -> Tuple[List[Any], int]:
    out: List[Any] = []
    while i < len(lines):
        cur_indent, text = lines[i]
        if cur_indent < indent:
            break
        if cur_indent != indent or not text.startswith("- "):
            break
        raw_value = text[2:].strip()
        if raw_value:
            out.append(_parse_scalar(raw_value))
            i += 1
        else:
            out_value, i = _parse_block(lines, i + 1, lines[i + 1][0])
            out.append(out_value)
    return out, i


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
