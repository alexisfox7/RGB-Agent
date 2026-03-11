"""Grid formatting, hashing, diffing, and component detection."""
from __future__ import annotations

import hashlib
from collections import defaultdict

_ASCII_PALETTE = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1EG[]?-_+~<>i!lI;:,\"^`'. "


def format_grid_ascii(grid: list[list[int]]) -> str:
    if not grid:
        return "(empty grid)"
    palette = _ASCII_PALETTE
    n = len(palette)
    lines = []
    for row in grid:
        chars = []
        for v in row:
            idx = min(int((max(0, min(15, int(v))) / 16) * (n - 1)), n - 1)
            chars.append(palette[idx])
        lines.append("".join(chars))
    return "\n".join(lines)


def hash_grid_state(grid: list[list[int]]) -> str:
    return hashlib.md5(str(grid).encode()).hexdigest()[:12]


def compute_grid_diff(old_grid: list, new_grid: list) -> str:
    if not old_grid or not new_grid:
        return "(no previous state)"
    groups: dict[tuple, list[str]] = defaultdict(list)
    for r, (old_row, new_row) in enumerate(zip(old_grid, new_grid)):
        for c, (old_val, new_val) in enumerate(zip(old_row, new_row)):
            if old_val != new_val:
                groups[(old_val, new_val)].append(f"({r},{c})")
    if not groups:
        return "(no change)"
    parts = []
    for (old_val, new_val), coords in sorted(groups.items()):
        parts.append(f"{old_val}->{new_val}: {', '.join(coords)}")
    return "; ".join(parts)


def find_connected_components(grid: list[list[int]]) -> dict[tuple, int]:
    """BFS flood-fill assigning a component ID to every cell."""
    if not grid:
        return {}
    rows, cols = len(grid), len(grid[0])
    comp_map: dict[tuple, int] = {}
    comp_id = 0

    def bfs(start_r: int, start_c: int, value: int) -> None:
        nonlocal comp_id
        queue = [(start_r, start_c)]
        while queue:
            r, c = queue.pop(0)
            if (r, c) in comp_map:
                continue
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if grid[r][c] != value:
                continue
            comp_map[(r, c)] = comp_id
            queue.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in comp_map:
                bfs(r, c, grid[r][c])
                comp_id += 1
    return comp_map


def get_click_info(grid: list[list[int]], row: int, col: int) -> tuple[str, str]:
    """Return (label, component_id) for the cell at (row, col)."""
    if not grid or row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]):
        return "?", "invalid"
    value = grid[row][col]
    comp_map = find_connected_components(grid)
    cid = comp_map.get((row, col), -1)
    size = sum(1 for v in comp_map.values() if v == cid)
    return f"val={value},comp_size={size}", f"val{value}_comp{cid}"
