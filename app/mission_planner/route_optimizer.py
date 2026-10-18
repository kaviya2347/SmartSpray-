"""
app/mission_planner/route_optimizer.py

Module 8 — spray route optimisation (pure functions, no I/O).

The drone starts and ends at a depot (the field origin / refill point) and
must visit every plant that needs spraying. This is a small Travelling
Salesman Problem (tens of plants), solved with:

    1. nearest-neighbour construction, then
    2. 2-opt local search (reverse segments while the tour gets shorter).

For 50-80 plants this runs in milliseconds and is typically within a few
percent of optimal. No external solver (OR-Tools) is required.

Baselines reported next to the optimised route:
    * id_order_m   visiting spray targets in plant_id order
    * blanket_m    lawn-mower coverage of the whole field (what a uniform
                   spraying pass flies)
"""

from __future__ import annotations

import math
from typing import Sequence

Point = tuple[float, float]


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tour_length(points: Sequence[Point], order: Sequence[int],
                depot: Point | None = None) -> float:
    """Length of depot -> points[order...] -> depot (open path if depot is None)."""
    if not order:
        return 0.0
    total = 0.0
    seq = [points[i] for i in order]
    if depot is not None:
        seq = [depot] + seq + [depot]
    for a, b in zip(seq[:-1], seq[1:]):
        total += dist(a, b)
    return total


def nearest_neighbour(points: Sequence[Point], depot: Point) -> list[int]:
    remaining = set(range(len(points)))
    order: list[int] = []
    cur = depot
    while remaining:
        # Deterministic tie-break on index.
        nxt = min(remaining, key=lambda i: (dist(cur, points[i]), i))
        order.append(nxt)
        remaining.remove(nxt)
        cur = points[nxt]
    return order


def two_opt(points: Sequence[Point], order: list[int], depot: Point,
            max_passes: int = 50) -> list[int]:
    """Classic 2-opt on a closed tour that starts/ends at the depot."""
    if len(order) < 3:
        return list(order)
    nodes = [depot] + [points[i] for i in order] + [depot]
    idx = [-1] + list(order) + [-1]
    improved, passes = True, 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, len(nodes) - 2):
            for k in range(i + 1, len(nodes) - 1):
                a, b = nodes[i - 1], nodes[i]
                c, d = nodes[k], nodes[k + 1]
                delta = (dist(a, c) + dist(b, d)) - (dist(a, b) + dist(c, d))
                if delta < -1e-9:
                    nodes[i:k + 1] = reversed(nodes[i:k + 1])
                    idx[i:k + 1] = reversed(idx[i:k + 1])
                    improved = True
    return idx[1:-1]


def optimise_route(points: Sequence[Point], depot: Point = (0.0, 0.0)) -> list[int]:
    """Nearest-neighbour + 2-opt. Returns the visiting order (indices)."""
    if not points:
        return []
    return two_opt(points, nearest_neighbour(points, depot), depot)


def lawnmower_length(width_m: float, height_m: float, swath_m: float,
                     depot: Point = (0.0, 0.0)) -> float:
    """
    Length of a boustrophedon (lawn-mower) pass covering a width x height
    field with the given spray swath, starting and ending at the depot.
    """
    if width_m <= 0 or height_m <= 0 or swath_m <= 0:
        return 0.0
    # 5 % tolerance: a field a few centimetres taller than a whole number of
    # swaths does not need an extra pass (swaths overlap slightly).
    n_lines = max(1, math.ceil(height_m / swath_m - 0.05))
    ys = [min(height_m, swath_m / 2 + i * swath_m) for i in range(n_lines)]
    path: list[Point] = [depot]
    for i, y in enumerate(ys):
        xs = (0.0, width_m) if i % 2 == 0 else (width_m, 0.0)
        path += [(xs[0], y), (xs[1], y)]
    path.append(depot)
    return sum(dist(a, b) for a, b in zip(path[:-1], path[1:]))


def split_into_sorties(order: list[int], demands: Sequence[float],
                       capacity: float) -> list[list[int]]:
    """
    Route-first, cluster-second: cut the optimised tour into consecutive
    sorties whose total demand fits in one tank.
    """
    if capacity <= 0:
        raise ValueError("tank capacity must be positive")
    sorties: list[list[int]] = []
    current: list[int] = []
    load = 0.0
    for i in order:
        need = float(demands[i])
        if need > capacity:
            raise ValueError(f"single plant dose {need} exceeds tank capacity {capacity}")
        if current and load + need > capacity + 1e-9:
            sorties.append(current)
            current, load = [], 0.0
        current.append(i)
        load += need
    if current:
        sorties.append(current)
    return sorties


def plan_capacitated_route(points: Sequence[Point], demands: Sequence[float],
                           capacity: float, depot: Point = (0.0, 0.0),
                           baseline_order: Sequence[int] | None = None) -> list[list[int]]:
    """
    Capacity-aware routing (a small CVRP heuristic):

      1. optimise one giant tour (nearest neighbour + 2-opt);
      2. try every rotation of that tour in both directions (plus an optional
         baseline order), cut each into tank-sized sorties, 2-opt every sortie;
      3. keep the candidate with the shortest total distance.

    Because the baseline order is one of the candidates, the result is never
    longer than the baseline.
    """
    if not points:
        return []
    tour = optimise_route(points, depot)
    candidates = []
    for seq in (tour, list(reversed(tour))):
        for k in range(len(seq)):
            candidates.append(seq[k:] + seq[:k])
    if baseline_order is not None:
        candidates.append(list(baseline_order))

    best, best_len = None, float("inf")
    for cand in candidates:
        chunks = [two_opt(points, c, depot)
                  for c in split_into_sorties(cand, demands, capacity)]
        total = sum(tour_length(points, c, depot) for c in chunks)
        if total < best_len - 1e-9:
            best, best_len = chunks, total
    return best


# ── Re-planning from the drone's current position ────────────────────────────

def path_length(points: Sequence[Point], order: Sequence[int], start: Point,
                end: Point | None) -> float:
    """Length of start -> points[order...] -> end (end=None: stop at last point)."""
    seq = [start] + [points[i] for i in order] + ([end] if end is not None else [])
    return sum(dist(a, b) for a, b in zip(seq[:-1], seq[1:]))


def two_opt_path(points: Sequence[Point], order: list[int], start: Point,
                 end: Point | None, max_passes: int = 50) -> list[int]:
    """2-opt on an open path with fixed start (and optional fixed end)."""
    if len(order) < 2:
        return list(order)
    nodes = [start] + [points[i] for i in order] + ([end] if end is not None else [])
    idx = [-1] + list(order) + ([-1] if end is not None else [])
    last = len(nodes) - (1 if end is not None else 0)
    improved, passes = True, 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, last - 1):
            for k in range(i + 1, last):
                a, b = nodes[i - 1], nodes[i]
                c = nodes[k]
                d = nodes[k + 1] if k + 1 < len(nodes) else None
                before = dist(a, b) + (dist(c, d) if d is not None else 0.0)
                after = dist(a, c) + (dist(b, d) if d is not None else 0.0)
                if after - before < -1e-9:
                    nodes[i:k + 1] = reversed(nodes[i:k + 1])
                    idx[i:k + 1] = reversed(idx[i:k + 1])
                    improved = True
    return [i for i in idx[1:] if i != -1]


def plan_from_position(points: Sequence[Point], demands: Sequence[float],
                       start: Point, tank_ml: float, depot: Point,
                       capacity: float) -> dict:
    """
    Re-plan after a manual override (or any interruption).

    The drone is at `start` with `tank_ml` of mix left. It first serves as
    many targets as the remaining tank allows (nearest-neighbour order, then
    2-opt on the open path start -> targets -> depot), flies home, and the
    rest is planned as normal capacity-aware sorties from the depot.

    Returns {"first": [indices], "later": [[indices], ...], "distance_m": float}
    """
    remaining = list(range(len(points)))
    first: list[int] = []
    if tank_ml > 1e-6 and remaining:
        cur, load = start, 0.0
        pool = set(remaining)
        while pool:
            nxt = min(pool, key=lambda i: (dist(cur, points[i]), i))
            if load + float(demands[nxt]) > tank_ml + 1e-6:
                # try any other target that still fits
                fits = [i for i in pool if load + float(demands[i]) <= tank_ml + 1e-6]
                if not fits:
                    break
                nxt = min(fits, key=lambda i: (dist(cur, points[i]), i))
            first.append(nxt)
            load += float(demands[nxt])
            pool.remove(nxt)
            cur = points[nxt]
        first = two_opt_path(points, first, start, depot)
    rest = [i for i in remaining if i not in set(first)]
    later: list[list[int]] = []
    if rest:
        sub_pts = [points[i] for i in rest]
        sub_dem = [demands[i] for i in rest]
        chunks = plan_capacitated_route(sub_pts, sub_dem, capacity, depot)
        later = [[rest[j] for j in chunk] for chunk in chunks]
    total = path_length(points, first, start, depot) if first else dist(start, depot)
    total += sum(tour_length(points, c, depot) for c in later)
    return {"first": first, "later": later, "distance_m": total}
