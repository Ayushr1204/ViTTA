"""
Unit tests for ViTTA tracking utilities, metrics, and resampler.

Run with:  python -m pytest tests/ -v
"""

import math
import numpy as np
import pytest

from vitta.class_names import CLASS_NAMES, class_name
from vitta.tracking.tracker_utils import (
    Detection,
    KalmanBoxTracker,
    iou_batch,
    associate_detections_to_tracks,
    compute_centroid,
    smooth_bbox,
    interpolate_bboxes,
)
from vitta.tracking.metrics import (
    euclidean_distance,
    compute_instantaneous_speed,
    compute_cumulative_distance,
    compute_cumulative_distances,
)
from vitta.tracking.resampler import TrackResampler, FrameRecord, ResampledRecord


# ═══════════════════════════════════════════════════════════════════════
#  class_names
# ═══════════════════════════════════════════════════════════════════════

class TestClassNames:
    def test_known_classes(self):
        assert class_name(0) == "Car"
        assert class_name(4) == "2W"
        assert class_name(7) == "Pedestrian"

    def test_unknown_class(self):
        assert class_name(99) == "cls99"

    def test_all_classes_defined(self):
        assert len(CLASS_NAMES) == 8


# ═══════════════════════════════════════════════════════════════════════
#  IoU
# ═══════════════════════════════════════════════════════════════════════

class TestIoU:
    def test_identical_boxes(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        result = iou_batch(a, a)
        assert result.shape == (1, 1)
        assert abs(result[0, 0] - 1.0) < 1e-6

    def test_no_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[20, 20, 30, 30]], dtype=np.float64)
        result = iou_batch(a, b)
        assert result[0, 0] == 0.0

    def test_partial_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[5, 5, 15, 15]], dtype=np.float64)
        result = iou_batch(a, b)
        # Intersection = 5*5=25, Union = 100+100-25=175
        expected = 25.0 / 175.0
        assert abs(result[0, 0] - expected) < 1e-6

    def test_multiple_boxes(self):
        a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float64)
        b = np.array([[5, 5, 15, 15]], dtype=np.float64)
        result = iou_batch(a, b)
        assert result.shape == (2, 1)
        assert result[0, 0] > 0  # partial overlap
        assert result[1, 0] == 0  # no overlap

    def test_empty_input(self):
        a = np.empty((0, 4), dtype=np.float64)
        b = np.array([[0, 0, 10, 10]], dtype=np.float64)
        result = iou_batch(a, b)
        assert result.shape == (0, 1)


# ═══════════════════════════════════════════════════════════════════════
#  Hungarian matching
# ═══════════════════════════════════════════════════════════════════════

class TestAssociation:
    def test_perfect_match(self):
        iou_matrix = np.array([[1.0]], dtype=np.float64)
        matches, unm_det, unm_trk = associate_detections_to_tracks(iou_matrix, 0.3)
        assert matches == [(0, 0)]
        assert unm_det == []
        assert unm_trk == []

    def test_below_threshold(self):
        iou_matrix = np.array([[0.1]], dtype=np.float64)
        matches, unm_det, unm_trk = associate_detections_to_tracks(iou_matrix, 0.3)
        assert matches == []
        assert unm_det == [0]
        assert unm_trk == [0]

    def test_empty_matrix(self):
        iou_matrix = np.empty((0, 0), dtype=np.float64)
        matches, unm_det, unm_trk = associate_detections_to_tracks(iou_matrix, 0.3)
        assert matches == []


# ═══════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ═══════════════════════════════════════════════════════════════════════

class TestGeometry:
    def test_centroid(self):
        bbox = np.array([0, 0, 10, 10])
        cx, cy = compute_centroid(bbox)
        assert cx == 5.0
        assert cy == 5.0

    def test_smooth_bbox_alpha_1(self):
        old = np.array([0, 0, 10, 10], dtype=np.float64)
        new = np.array([5, 5, 15, 15], dtype=np.float64)
        result = smooth_bbox(old, new, alpha=1.0)
        np.testing.assert_array_almost_equal(result, new)

    def test_smooth_bbox_alpha_0(self):
        old = np.array([0, 0, 10, 10], dtype=np.float64)
        new = np.array([5, 5, 15, 15], dtype=np.float64)
        result = smooth_bbox(old, new, alpha=0.0)
        np.testing.assert_array_almost_equal(result, old)

    def test_interpolate_bboxes(self):
        start = np.array([0, 0, 10, 10], dtype=np.float64)
        end = np.array([10, 10, 20, 20], dtype=np.float64)
        result = interpolate_bboxes(start, end, 1)
        assert len(result) == 1
        np.testing.assert_array_almost_equal(result[0], [5, 5, 15, 15])

    def test_interpolate_zero_frames(self):
        start = np.array([0, 0, 10, 10], dtype=np.float64)
        end = np.array([10, 10, 20, 20], dtype=np.float64)
        assert interpolate_bboxes(start, end, 0) == []


# ═══════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════

class TestMetrics:
    def test_euclidean_distance(self):
        assert euclidean_distance((0, 0), (3, 4)) == 5.0

    def test_euclidean_distance_same_point(self):
        assert euclidean_distance((5, 5), (5, 5)) == 0.0

    def test_speed_basic(self):
        speed = compute_instantaneous_speed((0, 0), (30, 40), 1.0)
        assert abs(speed - 50.0) < 1e-6

    def test_speed_zero_time(self):
        assert compute_instantaneous_speed((0, 0), (10, 10), 0.0) == 0.0

    def test_cumulative_distance(self):
        traj = [(0, 0), (3, 4), (6, 8)]
        dist = compute_cumulative_distance(traj)
        assert abs(dist - 10.0) < 1e-6  # 5 + 5

    def test_cumulative_distance_single_point(self):
        assert compute_cumulative_distance([(5, 5)]) == 0.0

    def test_cumulative_distances_running(self):
        traj = [(0, 0), (3, 4), (6, 8)]
        dists = compute_cumulative_distances(traj)
        assert len(dists) == 3
        assert dists[0] == 0.0
        assert abs(dists[1] - 5.0) < 1e-6
        assert abs(dists[2] - 10.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════════
#  Kalman filter
# ═══════════════════════════════════════════════════════════════════════

class TestKalmanBoxTracker:
    def setup_method(self):
        KalmanBoxTracker.reset_id_counter()

    def test_id_increments(self):
        t1 = KalmanBoxTracker(np.array([0, 0, 10, 10]))
        t2 = KalmanBoxTracker(np.array([20, 20, 30, 30]))
        assert t2.id == t1.id + 1

    def test_get_state_close_to_init(self):
        bbox = np.array([100, 200, 300, 400], dtype=np.float64)
        tracker = KalmanBoxTracker(bbox)
        state = tracker.get_state()
        np.testing.assert_array_almost_equal(state, bbox, decimal=0)

    def test_predict_update_cycle(self):
        bbox = np.array([100, 200, 150, 250], dtype=np.float64)
        tracker = KalmanBoxTracker(bbox)
        tracker.predict()
        tracker.update(bbox)
        assert tracker.hits == 2
        assert tracker.time_since_update == 0


# ═══════════════════════════════════════════════════════════════════════
#  Resampler
# ═══════════════════════════════════════════════════════════════════════

class TestResampler:
    def test_basic_resampling(self):
        resampler = TrackResampler(interval_sec=1.0)
        for i in range(90):  # 3 seconds at 30fps
            ts = i / 30.0
            resampler.add_record(FrameRecord(
                frame_id=i, timestamp=ts, track_id=1, class_id=0,
                confidence=0.9,
                bbox=(100 + i, 200 + i, 150 + i, 250 + i),
                centroid=(125.0 + i, 225.0 + i),
            ))

        results = resampler.resample()
        # Should have records at t=0, 1, 2
        assert len(results) == 3
        assert results[0].timestamp_sec == 0.0
        assert results[1].timestamp_sec == 1.0
        assert results[2].timestamp_sec == 2.0

    def test_speed_computation(self):
        resampler = TrackResampler(interval_sec=1.0)
        for i in range(90):
            ts = i / 30.0
            resampler.add_record(FrameRecord(
                frame_id=i, timestamp=ts, track_id=1, class_id=0,
                confidence=0.9,
                bbox=(100, 200, 150, 250),
                centroid=(125.0 + i, 225.0),  # moving 1px/frame = 30px/sec in x
            ))

        results = resampler.resample()
        # Speed at t=1s should be ~30 px/s (moved 30px in x over 1 second)
        assert results[1].speed_px_per_sec == pytest.approx(30.0, abs=0.5)

    def test_class_name_in_output(self):
        resampler = TrackResampler(interval_sec=1.0)
        for i in range(60):
            resampler.add_record(FrameRecord(
                frame_id=i, timestamp=i / 30.0, track_id=1, class_id=3,
                confidence=0.8, bbox=(0, 0, 10, 10), centroid=(5, 5),
            ))
        results = resampler.resample()
        assert results[0].class_name == "Auto"

    def test_multiple_tracks(self):
        resampler = TrackResampler(interval_sec=1.0)
        for i in range(60):
            ts = i / 30.0
            resampler.add_record(FrameRecord(
                frame_id=i, timestamp=ts, track_id=1, class_id=0,
                confidence=0.9, bbox=(0, 0, 10, 10), centroid=(5, 5),
            ))
            resampler.add_record(FrameRecord(
                frame_id=i, timestamp=ts, track_id=2, class_id=1,
                confidence=0.8, bbox=(20, 20, 30, 30), centroid=(25, 25),
            ))

        results = resampler.resample()
        track_ids = set(r.track_id for r in results)
        assert track_ids == {1, 2}
