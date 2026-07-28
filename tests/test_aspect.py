"""Aspect-ratio correction.

MediaPipe normalises x by frame *width* and y by frame *height*.  On a 16:9
webcam that stretches every horizontal measurement by 1.78x, which skews joint
angles, finger splay and pinch distance.  Left uncorrected it makes a closed
fist read as a thumbs-down on real hardware -- something synthetic tests using
square coordinates never reveal.
"""

from __future__ import annotations

import numpy as np
import pytest
from synth import make_state, pose

from gesturekit import geometry as geo
from gesturekit.gestures.continuous import PointerConfig, PointerRecognizer
from gesturekit.gestures.static import score_all

ASPECTS = [4 / 3, 16 / 9, 16 / 10, 21 / 9]
POSES = ["open_palm", "closed_fist", "peace", "point_up", "three", "four",
         "rock", "call_me", "thumbs_up", "pinch", "ok"]


def as_camera_reports(pts: np.ndarray, aspect: float) -> np.ndarray:
    """Squash x the way a real camera's normalisation does."""
    out = pts.copy()
    out[:, 0] = 0.5 + (out[:, 0] - 0.5) / aspect
    out[:, 2] = out[:, 2] / aspect
    return out


def test_correction_is_identity_for_square_frames():
    pts = pose("peace")
    np.testing.assert_array_equal(geo.correct_aspect(pts, 1.0), pts)


def test_correction_is_the_inverse_of_camera_normalisation():
    pts = pose("peace")
    for aspect in ASPECTS:
        recovered = geo.correct_aspect(as_camera_reports(pts, aspect), aspect)
        # Recovers the original shape up to a harmless x offset.
        centred = recovered[:, :2] - recovered[0, :2]
        original = pts[:, :2] - pts[0, :2]
        np.testing.assert_allclose(centred, original, atol=1e-5)


def test_correction_handles_empty_input():
    assert geo.correct_aspect(np.zeros((0, 3), np.float32), 1.78).shape == (0, 3)


@pytest.mark.parametrize("aspect", ASPECTS)
@pytest.mark.parametrize("name", POSES)
def test_poses_survive_real_camera_geometry(name, aspect):
    """The regression: recognition after correction must match the ideal case."""
    raw = as_camera_reports(pose(name), aspect)
    corrected = geo.correct_aspect(raw, aspect)
    scores = score_all(make_state(corrected))
    assert max(scores, key=scores.get) == name


def test_uncorrected_geometry_is_actually_wrong():
    """Guards the guard: if this ever passes, the correction is untested."""
    wrong = 0
    for name in POSES:
        raw = as_camera_reports(pose(name), 16 / 9)
        scores = score_all(make_state(raw))
        wrong += max(scores, key=scores.get) != name
    assert wrong > 0, "aspect distortion no longer breaks anything -- retune this test"


@pytest.mark.parametrize("aspect", ASPECTS)
def test_curls_match_after_correction(aspect):
    ideal = geo.finger_curl(pose("peace"))
    fixed = geo.finger_curl(geo.correct_aspect(as_camera_reports(pose("peace"), aspect), aspect))
    np.testing.assert_allclose(fixed, ideal, atol=1e-4)


@pytest.mark.parametrize("aspect", ASPECTS)
def test_pinch_distance_matches_after_correction(aspect):
    ideal = geo.pinch_distance(pose("pinch"))
    raw = as_camera_reports(pose("pinch"), aspect)
    assert geo.pinch_distance(geo.correct_aspect(raw, aspect)) == pytest.approx(ideal, abs=1e-4)


@pytest.mark.parametrize("aspect", [1.0, 4 / 3, 16 / 9])
def test_pointer_maps_the_active_region_onto_the_whole_screen(aspect):
    """Whatever the camera's aspect, the region edges must reach the screen edges.

    The cursor follows the *index fingertip*, so the region bounds are probed
    by placing that fingertip -- not the wrist -- at each edge.
    """
    from gesturekit.gestures.base import RecognizerContext

    cfg = PointerConfig(region_x0=0.2, region_x1=0.8, region_y0=0.2, region_y1=0.8)

    def tip_frame_x(center_x):
        """Where the fingertip lands, in frame fractions, for a given wrist x."""
        pts = geo.correct_aspect(
            as_camera_reports(pose("point_up", center=(center_x, 0.5)), aspect), aspect
        )
        return float(pts[geo.INDEX_TIP, 0]) / aspect

    def center_placing_tip_at(target_frame_x):
        """Solve for the wrist x that puts the fingertip on ``target_frame_x``.

        The pose's ``center`` is applied before the camera's non-square
        normalisation, so the two coordinate spaces differ by the aspect
        factor. Measuring the mapping empirically avoids getting that backwards.
        """
        a, b = 0.0, 1.0
        slope = (tip_frame_x(b) - tip_frame_x(a)) / (b - a)
        return a + (target_frame_x - tip_frame_x(a)) / slope

    def cursor_x_for(target_frame_x):
        """Place the fingertip at ``target_frame_x`` and read the cursor out."""
        center = center_placing_tip_at(target_frame_x)
        assert tip_frame_x(center) == pytest.approx(target_frame_x, abs=1e-3)

        rec = PointerRecognizer(cfg, aspect=aspect)
        ctx = RecognizerContext()
        events = []
        # Enough frames for the One Euro filter to converge on a held position;
        # in real use the hand has always been tracked for a while already.
        for i in range(40):
            pts = geo.correct_aspect(
                as_camera_reports(pose("point_up", center=(center, 0.5)), aspect), aspect
            )
            ctx.push(make_state(pts, timestamp=1000 + i * 0.033))
            events += rec.update(ctx)
        moves = [e for e in events if e.name == "cursor_move"]
        assert moves, "pointer produced no cursor events"
        return moves[-1].data["x"]

    assert cursor_x_for(0.20) == pytest.approx(0.0, abs=0.12)
    assert cursor_x_for(0.80) == pytest.approx(1.0, abs=0.12)
    assert cursor_x_for(0.50) == pytest.approx(0.5, abs=0.12)


def test_tracker_config_defaults_to_square():
    from gesturekit.tracker import TrackerConfig

    assert TrackerConfig().aspect == 1.0


def test_hud_maps_landmarks_back_to_image_space():
    from gesturekit.hud import HUD

    hud = HUD(aspect=16 / 9)
    # A point at aspect-corrected x=1.0 sits at the right edge of a 1280px frame.
    px = hud._to_pixels(np.array([[16 / 9, 0.5]]), 1280, 720)
    assert int(px[0][0]) == pytest.approx(1280, abs=2)
    assert int(px[0][1]) == pytest.approx(360, abs=2)
