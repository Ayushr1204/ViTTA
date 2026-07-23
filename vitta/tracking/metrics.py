"""
Speed and distance metrics for tracked objects.

All computations use pixel coordinates.  A future homography module
could convert these to real-world metres / km/h.
"""

from __future__ import annotations

import math
from typing import List, Tuple


def euclidean_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> float:
    """Euclidean distance between two 2-D points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def compute_instantaneous_speed(
    prev_centroid: Tuple[float, float],
    curr_centroid: Tuple[float, float],
    time_delta_sec: float,
) -> float:
    """
    Instantaneous speed between two consecutive observations.

    Args:
        prev_centroid: (cx, cy) at time t-1.
        curr_centroid: (cx, cy) at time t.
        time_delta_sec: Elapsed seconds between the two observations.

    Returns:
        Speed in pixels per second.  Returns 0.0 if time_delta is zero.
    """
    if time_delta_sec <= 0:
        return 0.0
    return euclidean_distance(prev_centroid, curr_centroid) / time_delta_sec


def compute_cumulative_distance(
    trajectory: List[Tuple[float, float]],
) -> float:
    """
    Total distance along a trajectory (sum of consecutive centroid hops).

    Args:
        trajectory: Ordered list of (cx, cy) centroids.

    Returns:
        Cumulative distance in pixels.
    """
    if len(trajectory) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(trajectory)):
        total += euclidean_distance(trajectory[i - 1], trajectory[i])
    return total


def compute_cumulative_distances(
    trajectory: List[Tuple[float, float]],
) -> List[float]:
    """
    Running cumulative distance at each point in the trajectory.

    Returns a list of the same length as *trajectory* where element i
    is the total distance from trajectory[0] to trajectory[i].
    """
    if not trajectory:
        return []
    distances = [0.0]
    for i in range(1, len(trajectory)):
        distances.append(
            distances[-1] + euclidean_distance(trajectory[i - 1], trajectory[i])
        )
    return distances
