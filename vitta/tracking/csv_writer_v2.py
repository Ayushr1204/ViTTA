"""
Aggregated CSV writer — one row per vehicle.

Collects per-frame tracking data during processing, then at the end
produces a single CSV row per unique track_id with:
  - Static metadata columns (id, class, direction, behaviour, etc.)
  - Repeating temporal columns every 0.5s (position, speed, accel, headway)
"""

from __future__ import annotations

import csv
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from vitta.class_names import class_name as get_class_name

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Per-frame observation stored during processing
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FrameObservation:
    """A single observation of a tracked vehicle in one frame."""
    frame_id: int
    timestamp: float        # seconds from video start
    cx: float               # centroid x (pixel coords, top-left origin)
    cy: float               # centroid y (pixel coords, top-left origin)
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float
    track_id: int
    rcx: float = 0.0        # rectified centroid x (perspective-corrected)
    rcy: float = 0.0        # rectified centroid y (perspective-corrected)


# ═══════════════════════════════════════════════════════════════════════
# Aggregated CSV writer
# ═══════════════════════════════════════════════════════════════════════

class AggregatedCSVWriter:
    """
    Collects frame-by-frame observations, then writes one row per
    vehicle to a CSV when finalise() is called.

    Usage::

        writer = AggregatedCSVWriter(path, fps=30, frame_height=1080)
        # during processing:
        writer.record_frame(frame_id, timestamp, tracked_objects)
        # at the end:
        writer.finalise()
    """

    SAMPLE_INTERVAL_SEC = 0.5  # temporal sampling interval
    MIN_OBS_FOR_VALID_TRACK = 5  # discard tracks with fewer observations (spurious)

    def __init__(
        self,
        output_path: str | Path,
        fps: float = 30.0,
        frame_height: int = 1080,
        pixels_per_metre: Optional[float] = None,
        roi_points: Optional[List[List[float]]] = None,
        calib_points: Optional[List[List[float]]] = None,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_height = frame_height
        self.pixels_per_metre = pixels_per_metre
        self.is_calibrated = pixels_per_metre is not None and pixels_per_metre > 0

        # track_id → list of FrameObservation (time-sorted)
        self._observations: Dict[int, List[FrameObservation]] = defaultdict(list)
        self._rows_written = 0

        # Perspective correction via homography
        # NOTE: Homography-based rectification is currently disabled.
        # Mapping an arbitrary ROI trapezoid to a fixed rectangle can
        # produce extreme coordinate distortion near vanishing points.
        # A proper implementation would need vanishing-point estimation
        # or multi-point ground-truth calibration.
        self._homography: Optional[np.ndarray] = None
        self._rectified_ppm: Optional[float] = None
        self._rect_height: float = 1000.0
        self._has_perspective = False

        logger.info(
            f"AggregatedCSVWriter opened: {self.output_path} "
            f"(fps={fps}, height={frame_height}, ppm={pixels_per_metre}, "
            f"calibrated={self.is_calibrated}, perspective={self._has_perspective})"
        )

    # ── Perspective correction ─────────────────────────────────────────

    def _setup_homography(
        self,
        roi_points: List[List[float]],
        calib_points: List[List[float]],
    ) -> None:
        """
        Compute a perspective transform (homography) from the ROI
        quadrilateral to a top-down rectified rectangle.

        This corrects for perspective distortion: objects farther from the
        camera occupy fewer pixels per real-world metre.  By rectifying
        all centroids into a bird's-eye coordinate space, distance and
        speed calculations become perspective-invariant.
        """
        try:
            src = np.array(roi_points[:4], dtype=np.float32)
            ordered = self._order_quad_points(src)

            # Map to a 1000×1000 rectified rectangle
            dst = np.array(
                [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
                dtype=np.float32,
            )
            H = cv2.getPerspectiveTransform(ordered, dst)

            # Transform calibration points into rectified space
            ca = np.array([[[calib_points[0][0], calib_points[0][1]]]],
                          dtype=np.float32)
            cb = np.array([[[calib_points[1][0], calib_points[1][1]]]],
                          dtype=np.float32)
            ca_r = cv2.perspectiveTransform(ca, H)[0][0]
            cb_r = cv2.perspectiveTransform(cb, H)[0][0]

            rect_dist = math.sqrt(
                (ca_r[0] - cb_r[0]) ** 2 + (ca_r[1] - cb_r[1]) ** 2
            )

            # Derive real-world distance from the original calibration
            orig_dist = math.sqrt(
                (calib_points[1][0] - calib_points[0][0]) ** 2
                + (calib_points[1][1] - calib_points[0][1]) ** 2
            )
            road_length_m = orig_dist / self.pixels_per_metre

            self._rectified_ppm = rect_dist / road_length_m
            self._homography = H
            self._has_perspective = True

            logger.info(
                f"Perspective correction enabled: "
                f"original px/m={self.pixels_per_metre:.2f}, "
                f"rectified px/m={self._rectified_ppm:.2f}, "
                f"road_length={road_length_m:.2f}m"
            )
        except Exception as exc:
            logger.warning(
                f"Failed to compute homography, falling back to flat "
                f"calibration: {exc}"
            )
            self._homography = None
            self._has_perspective = False

    @staticmethod
    def _order_quad_points(pts: np.ndarray) -> np.ndarray:
        """
        Order 4 points as [TL, TR, BR, BL] for cv2.getPerspectiveTransform.

        Uses sum (x+y) and difference (y-x) heuristics.
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()
        rect[0] = pts[np.argmin(s)]   # TL: smallest x+y
        rect[2] = pts[np.argmax(s)]   # BR: largest  x+y
        rect[1] = pts[np.argmin(d)]   # TR: smallest y-x
        rect[3] = pts[np.argmax(d)]   # BL: largest  y-x
        return rect

    def _rectify_point(self, cx: float, cy: float) -> Tuple[float, float]:
        """Transform a point from image space to rectified (bird's-eye) space."""
        if self._homography is None:
            return cx, cy
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        rp = cv2.perspectiveTransform(pt, self._homography)[0][0]
        return float(rp[0]), float(rp[1])

    # ── Unit helpers ──────────────────────────────────────────────────

    def _effective_ppm(self) -> float:
        """Return the pixels-per-metre for the active coordinate space."""
        if self._has_perspective and self._rectified_ppm:
            return self._rectified_ppm
        return self.pixels_per_metre or 1.0

    def _px_to_m(self, px_value: float) -> float:
        """Convert a pixel distance/speed to metres (or m/s)."""
        if self.is_calibrated:
            return px_value / self._effective_ppm()
        return px_value

    def _px_to_m_accel(self, px_accel: float) -> float:
        """Convert pixel-based acceleration (px/s²) to m/s²."""
        if self.is_calibrated:
            return px_accel / self._effective_ppm()
        return px_accel

    def _unit_suffix(self, px_unit: str, m_unit: str) -> str:
        """Return the correct unit label."""
        return m_unit if self.is_calibrated else px_unit

    # ── Recording (called per frame during processing) ────────────────

    def record_frame(
        self,
        frame_id: int,
        timestamp: float,
        tracked_objects,
    ) -> None:
        """
        Record observations for all tracked objects in a frame.

        Args:
            frame_id:        Integer frame index.
            timestamp:       Frame timestamp in seconds.
            tracked_objects: List of TrackedObject from ByteTracker.
        """
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            cx, cy = obj.centroid
            rcx, rcy = self._rectify_point(cx, cy)
            obs = FrameObservation(
                frame_id=frame_id,
                timestamp=timestamp,
                cx=cx, cy=cy,
                x1=x1, y1=y1, x2=x2, y2=y2,
                class_id=obj.class_id,
                confidence=obj.confidence,
                track_id=obj.track_id,
                rcx=rcx, rcy=rcy,
            )
            self._observations[obj.track_id].append(obs)

    # ── Finalisation (called once after processing is complete) ────────

    def finalise(self) -> None:
        """Aggregate all observations and write the CSV."""
        if not self._observations:
            logger.warning("No observations to write.")
            return

        # Filter out spurious short-lived tracks (ghost detections that
        # briefly confirm then vanish — not real vehicles).
        valid_observations = {
            tid: obs_list
            for tid, obs_list in self._observations.items()
            if len(obs_list) >= self.MIN_OBS_FOR_VALID_TRACK
        }
        n_filtered = len(self._observations) - len(valid_observations)
        if n_filtered > 0:
            logger.info(
                f"Filtered {n_filtered} spurious tracks "
                f"(< {self.MIN_OBS_FOR_VALID_TRACK} observations). "
                f"Keeping {len(valid_observations)} valid vehicles."
            )

        if not valid_observations:
            logger.warning("No valid tracks after filtering.")
            return

        # ── First pass: compute track summaries (WITHOUT behaviour) ───
        # Behaviour requires global statistics across ALL vehicles,
        # so we compute it in a separate pass after all summaries exist.
        track_summaries = {}
        for tid, obs_list in valid_observations.items():
            summary = self._compute_track_summary(tid, obs_list)
            track_summaries[tid] = summary

        # ── Second pass: compute temporal samples ────────────────────
        max_samples = 0
        all_sampled: Dict[int, List[dict]] = {}
        for tid, obs_list in valid_observations.items():
            samples = self._temporal_samples(obs_list, valid_observations)
            all_sampled[tid] = samples
            max_samples = max(max_samples, len(samples))

        # ── Third pass: behaviour classification (deferred) ──────────
        # Uses avg_speed from ALL vehicles to compute IQR thresholds,
        # and speed_cv from the sub-sampled temporal data (not noisy
        # per-frame data).
        all_avg_speeds: List[float] = []
        for tid, summary in track_summaries.items():
            all_avg_speeds.append(summary["avg_speed"])

            # Recompute speed_cv from the sub-sampled temporal speeds
            # instead of from noisy per-frame data.
            sampled_speeds = [
                s["inst_speed"] for s in all_sampled[tid]
                if s["inst_speed"] > 0
            ]
            if len(sampled_speeds) >= 2:
                mean_s = sum(sampled_speeds) / len(sampled_speeds)
                if mean_s > 0:
                    std_s = math.sqrt(
                        sum((s - mean_s) ** 2 for s in sampled_speeds)
                        / len(sampled_speeds)
                    )
                    summary["speed_cv"] = std_s / mean_s
                else:
                    summary["speed_cv"] = 0.0
            else:
                summary["speed_cv"] = 0.0

        # Compute global speed stats for behaviour classification
        sorted_speeds = sorted(all_avg_speeds)
        n = len(sorted_speeds)
        if n >= 4:
            q1 = sorted_speeds[n // 4]
            q3 = sorted_speeds[3 * n // 4]
            iqr = q3 - q1
            median_speed = sorted_speeds[n // 2]
        else:
            q1 = sorted_speeds[0] if sorted_speeds else 0
            q3 = sorted_speeds[-1] if sorted_speeds else 0
            iqr = q3 - q1
            median_speed = sum(sorted_speeds) / n if n > 0 else 0

        # Classify behaviour for each vehicle
        for tid, summary in track_summaries.items():
            # Compute kinematic indicators from temporal samples
            samples = all_sampled[tid]
            lat_accels = [s["accel_lateral"] for s in samples]
            lin_accels = [s["accel_linear"] for s in samples]
            time_headways = [s["time_headway"] for s in samples if s["time_headway"] > 0]

            avg_lateral_accel_raw = (
                sum(abs(a) for a in lat_accels) / len(lat_accels)
                if lat_accels else 0.0
            )
            # Use 5th percentile instead of raw min() — a single noisy
            # frame can produce extreme fake deceleration that would
            # flag every vehicle as "Aggressive Braking".
            if len(lin_accels) >= 3:
                sorted_la = sorted(lin_accels)
                p5_idx = max(0, int(len(sorted_la) * 0.05))
                min_linear_accel_raw = sorted_la[p5_idx]
            else:
                min_linear_accel_raw = min(lin_accels) if lin_accels else 0.0
            avg_time_headway = (
                sum(time_headways) / len(time_headways)
                if time_headways else 999.0
            )

            # Convert accelerations to m/s² for meaningful thresholds.
            # When uncalibrated, set to 0 so kinematic classes are skipped
            # (px/s² thresholds are not physically meaningful).
            if self.is_calibrated:
                avg_lat_accel = self._px_to_m_accel(avg_lateral_accel_raw)
                min_lin_accel = self._px_to_m_accel(min_linear_accel_raw)
            else:
                avg_lat_accel = 0.0
                min_lin_accel = 0.0

            behaviour = self._classify_behaviour(
                summary["avg_speed"],
                summary["speed_cv"],
                median_speed, q1, q3, iqr,
                avg_lateral_accel=avg_lat_accel,
                min_linear_accel=min_lin_accel,
                avg_time_headway=avg_time_headway,
            )
            summary["behavior_class"] = behaviour

        # ── Compute total_distance from temporal samples ────────────
        # total_distance = sum of all distance_travelled values from
        # the temporal samples.  This is computed AFTER the vehicle
        # exits the frame, so we have all the data.
        for tid, summary in track_summaries.items():
            samples = all_sampled[tid]
            total_dist = sum(s["distance_travelled"] for s in samples)
            # Convert to metres if calibrated (distance_travelled is in px)
            summary["total_distance"] = total_dist

        # ── Build column headers ─────────────────────────────────────
        speed_unit = self._unit_suffix("px_per_s", "m_per_s")
        dist_unit = self._unit_suffix("px", "m")
        accel_unit = self._unit_suffix("px_per_s2", "m_per_s2")

        static_cols = [
            "track_id", "timestamp_first_seen", "frame_id_first_seen",
            "class_id", "class_name", "confidence",
            f"avg_speed_{speed_unit}", "direction", "behavior_class",
            f"total_distance_{dist_unit}",
        ]
        repeating_cols_per_sample = [
            f"x_{dist_unit}", f"y_{dist_unit}",
            "trajectory_deg", f"inst_speed_{speed_unit}",
            f"distance_travelled_{dist_unit}",
            f"accel_linear_{accel_unit}", f"accel_lateral_{accel_unit}",
            f"dist_headway_{dist_unit}", f"time_headway_s",
        ]
        header = list(static_cols)
        for i in range(max_samples):
            for col in repeating_cols_per_sample:
                header.append(f"t{i}_{col}")

        # ── Write CSV ────────────────────────────────────────────────
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for tid in sorted(track_summaries.keys()):
                summary = track_summaries[tid]
                samples = all_sampled[tid]

                # Convert values to metres if calibrated
                avg_speed_out = self._px_to_m(summary["avg_speed"])
                total_dist_out = self._px_to_m(summary["total_distance"])

                row = [
                    tid,
                    f"{summary['first_timestamp']:.4f}",
                    summary["first_frame_id"],
                    summary["class_id"],
                    summary["class_name"],
                    f"{summary['avg_confidence']:.4f}",
                    f"{avg_speed_out:.2f}",
                    summary["direction"],
                    summary["behavior_class"],
                    f"{total_dist_out:.2f}",
                ]

                # Append temporal samples (pad with empty if fewer than max)
                for i in range(max_samples):
                    if i < len(samples):
                        s = samples[i]
                        # Convert to metres if calibrated
                        x_out = self._px_to_m(s["x"])
                        y_out = self._px_to_m(s["y"])
                        speed_out = self._px_to_m(s["inst_speed"])
                        dist_trav_out = self._px_to_m(s["distance_travelled"])
                        al_out = self._px_to_m_accel(s["accel_linear"])
                        at_out = self._px_to_m_accel(s["accel_lateral"])
                        dh_out = self._px_to_m(s["dist_headway"])

                        row.extend([
                            f"{x_out:.2f}",
                            f"{y_out:.2f}",
                            f"{s['trajectory_deg']:.1f}",
                            f"{speed_out:.2f}",
                            f"{dist_trav_out:.2f}",
                            f"{al_out:.2f}",
                            f"{at_out:.2f}",
                            f"{dh_out:.2f}",
                            f"{s['time_headway']:.2f}",
                        ])
                    else:
                        row.extend([""] * len(repeating_cols_per_sample))

                writer.writerow(row)
                self._rows_written += 1

        logger.info(
            f"AggregatedCSVWriter finalised: {self.output_path} "
            f"({self._rows_written} vehicle rows, "
            f"calibrated={self.is_calibrated})"
        )

    # ── Track summary computation ─────────────────────────────────────

    # Maximum plausible vehicle speed (px/s when uncalibrated, m/s when calibrated)
    # 250 km/h ≈ 69.4 m/s — generous cap for Indian traffic
    MAX_PLAUSIBLE_SPEED_MS = 70.0

    def _compute_track_summary(
        self, tid: int, obs_list: List[FrameObservation]
    ) -> dict:
        """Compute static metadata for one track."""
        obs_list.sort(key=lambda o: o.timestamp)

        first = obs_list[0]
        last = obs_list[-1]
        total_time = last.timestamp - first.timestamp

        # ── Compute per-segment speeds for robust median ─────────────
        # Using rectified coordinates (rcx/rcy) for perspective-correct
        # distances.  When no homography is active, rcx/rcy == cx/cy.
        segment_speeds: List[float] = []
        for i in range(1, len(obs_list)):
            dx = obs_list[i].rcx - obs_list[i - 1].rcx
            dy = obs_list[i].rcy - obs_list[i - 1].rcy
            seg = math.sqrt(dx * dx + dy * dy)
            dt = obs_list[i].timestamp - obs_list[i - 1].timestamp
            if dt > 0:
                segment_speeds.append(seg / dt)

        # Displacement for direction
        dx_total = last.rcx - first.rcx
        dy_total = last.rcy - first.rcy
        displacement = math.sqrt(dx_total * dx_total + dy_total * dy_total)

        # Average speed = median of segment speeds (robust to outliers)
        # then capped to physical plausibility
        if segment_speeds:
            sorted_seg = sorted(segment_speeds)
            mid = len(sorted_seg) // 2
            if len(sorted_seg) % 2 == 0 and len(sorted_seg) >= 2:
                median_speed = (sorted_seg[mid - 1] + sorted_seg[mid]) / 2
            else:
                median_speed = sorted_seg[mid]
            avg_speed = median_speed
        else:
            avg_speed = displacement / total_time if total_time > 0 else 0.0

        # Cap to physical plausibility
        max_speed_px = self._max_plausible_speed_px()
        avg_speed = min(avg_speed, max_speed_px)

        # ── Total distance ────────────────────────────────────────────
        # total_distance is now computed in finalise() as the sum of
        # per-sample distance_travelled values.  This ensures it is
        # only filled after the vehicle exits the frame and we have
        # all the data.  Set to 0 as placeholder here.
        total_dist = 0.0

        # speed_cv is computed later from sub-sampled temporal speeds
        # (set to 0 as placeholder; overwritten in finalise())
        speed_cv = 0.0

        # Direction (from first to last centroid)
        direction = self._compute_direction(dx_total, dy_total)

        # Average confidence
        avg_conf = sum(o.confidence for o in obs_list) / len(obs_list)

        return {
            "first_timestamp": first.timestamp,
            "first_frame_id": first.frame_id,
            "class_id": first.class_id,
            "class_name": get_class_name(first.class_id),
            "avg_confidence": avg_conf,
            "avg_speed": avg_speed,
            "total_distance": total_dist,
            "direction": direction,
            "speed_cv": speed_cv,
        }

    def _max_plausible_speed_px(self) -> float:
        """Return the maximum plausible speed in the active coordinate space (px/s)."""
        if self.is_calibrated:
            return self.MAX_PLAUSIBLE_SPEED_MS * self._effective_ppm()
        # Uncalibrated: assume ~2000 px/s as a generous cap
        # (a vehicle crossing 1920px in 1 second)
        return 2000.0

    # ── Temporal sampling ─────────────────────────────────────────────

    def _temporal_samples(
        self,
        obs_list: List[FrameObservation],
        all_observations: Dict[int, List[FrameObservation]],
    ) -> List[dict]:
        """
        Sample the track at 0.5s intervals. For each sample, compute
        position (bottom-left origin), trajectory, speed, acceleration,
        and headway.

        Acceleration convention (vehicle-frame):
        - accel_linear:      d(speed)/dt — +ve = speeding up, -ve = braking
        - accel_lateral:    speed × d(heading)/dt — +ve = turning right,
                            -ve = turning left (from vehicle's perspective)
        """
        if not obs_list:
            return []

        obs_list.sort(key=lambda o: o.timestamp)
        t_start = obs_list[0].timestamp
        t_end = obs_list[-1].timestamp

        # Pre-compute speeds for each observation for use in headway
        obs_speeds = self._precompute_speeds(obs_list)

        samples = []
        t = t_start
        prev_speed: Optional[float] = None
        prev_heading: Optional[float] = None
        prev_t: Optional[float] = None
        prev_x: Optional[float] = None
        prev_y: Optional[float] = None

        while t <= t_end + 1e-6:
            # Find nearest observation to time t
            nearest_idx = min(
                range(len(obs_list)),
                key=lambda i: abs(obs_list[i].timestamp - t),
            )
            nearest = obs_list[nearest_idx]

            # Position in rectified space
            x_out = nearest.rcx
            y_out = nearest.rcy
            if not self._has_perspective:
                # No homography — use bottom-left origin convention
                y_out = self.frame_height - nearest.cy

            # ── Distance travelled since last sample ─────────────────
            # Measures how far the vehicle moved from the last known
            # record.  Starts at 0 for the first record.
            if prev_x is not None and prev_y is not None:
                dx_dt = x_out - prev_x
                dy_dt = y_out - prev_y
                distance_travelled = math.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
            else:
                distance_travelled = 0.0

            # ── Trajectory heading (degrees, 0=East, 90=North) ───────
            # Use forward-looking for first obs so heading is never 0° by default
            heading = 0.0
            if nearest_idx > 0:
                dx = nearest.rcx - obs_list[nearest_idx - 1].rcx
                dy = -(nearest.rcy - obs_list[nearest_idx - 1].rcy)
                heading = math.degrees(math.atan2(dy, dx))
            elif nearest_idx == 0 and len(obs_list) > 1:
                # Forward-looking heading for the very first observation
                dx = obs_list[1].rcx - nearest.rcx
                dy = -(obs_list[1].rcy - nearest.rcy)
                heading = math.degrees(math.atan2(dy, dx))

            # Instantaneous speed (use pre-computed)
            inst_speed = obs_speeds[nearest_idx]

            # ── Linear acceleration: d(speed)/dt ─────────────────────
            # +ve = vehicle speeding up, -ve = braking
            accel_linear = 0.0
            if prev_speed is not None and prev_t is not None:
                dt_sample = t - prev_t
                if dt_sample > 0:
                    accel_linear = (inst_speed - prev_speed) / dt_sample

            # ── Lateral acceleration: speed × ω ──────────────────────
            # ω = d(heading)/dt in rad/s
            # +ve = turning right (clockwise from vehicle's POV)
            # -ve = turning left
            accel_lateral = 0.0
            if prev_heading is not None and prev_t is not None:
                dt_sample = t - prev_t
                dheading = heading - prev_heading
                # Normalize angle difference to [-180, 180]
                while dheading > 180:
                    dheading -= 360
                while dheading < -180:
                    dheading += 360
                if dt_sample > 0 and inst_speed > 0:
                    omega = math.radians(dheading) / dt_sample
                    # Convention: positive omega (counter-clockwise in math)
                    # corresponds to turning LEFT.  We want +ve = right turn,
                    # so negate.
                    accel_lateral = -inst_speed * omega

            # ── Headway ──────────────────────────────────────────────
            dist_hw, time_hw = self._compute_headway(
                current_obs=nearest,
                heading_deg=heading,
                sample_speed=inst_speed,
                all_observations=all_observations,
            )

            samples.append({
                "x": x_out,
                "y": y_out,
                "trajectory_deg": heading,
                "inst_speed": inst_speed,
                "distance_travelled": distance_travelled,
                "accel_linear": accel_linear,
                "accel_lateral": accel_lateral,
                "dist_headway": dist_hw,
                "time_headway": time_hw,
            })

            prev_speed = inst_speed
            prev_heading = heading
            prev_t = t
            prev_x = x_out
            prev_y = y_out
            t += self.SAMPLE_INTERVAL_SEC

        return samples

    # ── Speed pre-computation ─────────────────────────────────────────

    def _precompute_speeds(
        self, obs_list: List[FrameObservation]
    ) -> List[float]:
        """
        Pre-compute instantaneous speed for each observation.
        Uses forward-looking speed for the first observation so we
        don't get 0.0 at the start.

        Applies a 3-point median filter to remove single-frame jitter
        spikes, then caps to a physical maximum.
        """
        n = len(obs_list)
        raw_speeds = [0.0] * n

        # Backward-looking speed for obs[1..n-1]
        for i in range(1, n):
            dx = obs_list[i].rcx - obs_list[i - 1].rcx
            dy = obs_list[i].rcy - obs_list[i - 1].rcy
            seg = math.sqrt(dx * dx + dy * dy)
            dt = obs_list[i].timestamp - obs_list[i - 1].timestamp
            if dt > 0:
                raw_speeds[i] = seg / dt

        # Forward-looking speed for obs[0] (avoids 0.0 at start)
        if n >= 2:
            dx = obs_list[1].rcx - obs_list[0].rcx
            dy = obs_list[1].rcy - obs_list[0].rcy
            seg = math.sqrt(dx * dx + dy * dy)
            dt = obs_list[1].timestamp - obs_list[0].timestamp
            if dt > 0:
                raw_speeds[0] = seg / dt

        # ── Median filter (window=3) to suppress single-frame jitter ──
        speeds = list(raw_speeds)
        if n >= 3:
            for i in range(1, n - 1):
                triple = sorted([raw_speeds[i - 1], raw_speeds[i], raw_speeds[i + 1]])
                speeds[i] = triple[1]  # median of 3

        # ── Cap to physical maximum ──────────────────────────────────
        max_spd = self._max_plausible_speed_px()
        for i in range(n):
            if speeds[i] > max_spd:
                speeds[i] = max_spd

        return speeds

    # ── Headway computation ───────────────────────────────────────────

    def _compute_headway(
        self,
        current_obs: FrameObservation,
        heading_deg: float,
        sample_speed: float,
        all_observations: Dict[int, List[FrameObservation]],
    ) -> Tuple[float, float]:
        """
        Find the nearest vehicle ahead (in the direction of travel)
        at the same timestamp, and compute distance and time headway.

        Args:
            current_obs:      The observation at the current sample time.
            heading_deg:      Heading in degrees (0=East, 90=North) — must
                              already be valid (not a 0° default).
            sample_speed:     The instantaneous speed at this sample (px/s).
            all_observations: All vehicle observation lists (for finding
                              neighbours).

        Returns:
            (distance_headway_px, time_headway_s).
            Both distances in pixels — caller converts to metres.
            Returns (0, 0) when no vehicle is found ahead.
        """
        heading_rad = math.radians(heading_deg)
        dir_x = math.cos(heading_rad)
        dir_y = -math.sin(heading_rad)  # pixel coords (y increases downward)

        min_dist = float("inf")
        best_time_hw = 0.0

        for tid, obs_list in all_observations.items():
            if tid == current_obs.track_id:
                continue

            # Find nearest observation in time for this other vehicle
            best_other = None
            best_dt = float("inf")
            for o in obs_list:
                dt_abs = abs(o.timestamp - current_obs.timestamp)
                if dt_abs < best_dt:
                    best_dt = dt_abs
                    best_other = o
                elif dt_abs > best_dt + 1.0:
                    break  # obs_list is sorted by time

            # Wider time tolerance: 1.0 s (was 0.3 s)
            if best_other is None or best_dt > 1.0:
                continue

            # Vector from current to other vehicle (rectified coords)
            dx = best_other.rcx - current_obs.rcx
            dy = best_other.rcy - current_obs.rcy

            # Project onto heading direction (dot product)
            proj = dx * dir_x + dy * dir_y
            if proj <= 0:
                continue  # behind us, not ahead

            dist = math.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                min_dist = dist

                # Time headway = distance / speed of current vehicle
                if sample_speed > 0.5:
                    best_time_hw = dist / sample_speed
                else:
                    best_time_hw = 0.0  # stationary → headway meaningless

        if min_dist == float("inf"):
            return (0.0, 0.0)  # no vehicle ahead

        # Cap time headway at 120 s
        if best_time_hw > 120.0:
            best_time_hw = 120.0

        return (min_dist, best_time_hw)

    # ── Direction classification ──────────────────────────────────────

    @staticmethod
    def _compute_direction(dx: float, dy: float) -> str:
        """Classify direction from displacement (pixel coords, y-down)."""
        if abs(dx) < 1 and abs(dy) < 1:
            return "Stationary"
        # atan2 with inverted y for natural direction
        angle = math.degrees(math.atan2(-dy, dx))  # -dy because y-down
        if -45 <= angle < 45:
            return "Eastbound"
        elif 45 <= angle < 135:
            return "Northbound"
        elif -135 <= angle < -45:
            return "Southbound"
        else:
            return "Westbound"

    # ── Behaviour classification ──────────────────────────────────────

    @staticmethod
    def _classify_behaviour(
        avg_speed: float,
        speed_cv: float,
        median_speed: float,
        q1: float,
        q3: float,
        iqr: float,
        avg_lateral_accel: float = 0.0,
        min_linear_accel: float = 0.0,
        avg_time_headway: float = 999.0,
    ) -> str:
        """
        Classify vehicle behaviour using IQR-based outlier detection
        combined with kinematic indicators.

        Kinematic thresholds are in m/s² (caller must convert before
        calling).  When uncalibrated, caller passes 0.0 for accel
        values so kinematic classes are skipped.

        Priority order (first match wins):
        1. Stopped/Idling:      avg_speed near zero
        2. Speeding:            avg_speed above Q3 + 1.5×IQR
        3. Aggressive Braking:  peak deceleration < -8 m/s²
        4. Tailgating:          avg time headway < 1.2 s
        5. Lane Weaving:        avg |lateral accel| > 2 m/s²
        6. Slow:                avg_speed below Q1 - 1.5×IQR
        7. Erratic:             speed CV > 1.0
        8. Disciplined:         everything else
        """
        # 1. Stopped/Idling — vehicle barely moved
        if avg_speed < 0.5 and median_speed > 0:
            return "Stopped/Idling"

        if median_speed <= 0:
            return "Disciplined"

        upper_fence = q3 + 1.5 * iqr
        lower_fence = q1 - 1.5 * iqr

        # 2. Speeding — speed outliers take priority
        if avg_speed > upper_fence and upper_fence > 0:
            return "Speeding"

        # 3. Aggressive Braking — peak deceleration beyond -8 m/s²
        #    (only fires when calibrated; uncalibrated passes 0.0)
        if min_linear_accel < -8.0:
            return "Aggressive Braking"

        # 4. Tailgating — dangerously low average time headway
        if 0 < avg_time_headway < 1.2:
            return "Tailgating"

        # 5. Lane Weaving — high lateral acceleration indicates lane changes
        #    (only fires when calibrated; uncalibrated passes 0.0)
        if abs(avg_lateral_accel) > 2.0:
            return "Lane Weaving"

        # 6. Slow
        if avg_speed < lower_fence and lower_fence > 0:
            return "Slow"

        # 7. Erratic — highly variable speed (CV > 1.0)
        if speed_cv > 1.0:
            return "Erratic"

        return "Disciplined"

    # ── Properties ────────────────────────────────────────────────────

    @property
    def rows_written(self) -> int:
        """Number of vehicle rows written (available after finalise)."""
        return self._rows_written

    @property
    def total_observations(self) -> int:
        """Total frame-level observations recorded."""
        return sum(len(v) for v in self._observations.values())

    @property
    def unique_tracks(self) -> int:
        """Number of unique valid vehicle tracks (above min-observation threshold)."""
        return sum(
            1 for obs_list in self._observations.values()
            if len(obs_list) >= self.MIN_OBS_FOR_VALID_TRACK
        )
