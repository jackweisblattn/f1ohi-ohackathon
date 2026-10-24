"""Network-free unit tests.

These exercise the pure-Python logic (resampling, analysis math, figure
building, error-message formatting) without importing FastF1 or touching the
network, so they run anywhere.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from data import analysis  # noqa: E402
from data.loader import (  # noqa: E402
    DriverInfo,
    SessionData,
    _friendly_error,
)
from viz.track_map import build_track_figure  # noqa: E402


def _fake_session_data() -> SessionData:
    """Build a tiny synthetic session for figure/rendering tests."""
    t = np.arange(0.0, 5.0, 0.5)
    drivers = {
        "VER": DriverInfo("VER", "1", "Max Verstappen", "Red Bull", "#3671C6"),
        "HAM": DriverInfo("HAM", "44", "Lewis Hamilton", "Mercedes", "#27F4D2"),
    }
    positions = {
        "VER": pd.DataFrame({"X": np.sin(t), "Y": np.cos(t)}, index=t),
        "HAM": pd.DataFrame({"X": np.cos(t), "Y": np.sin(t)}, index=t),
    }
    outline = pd.DataFrame({"X": np.sin(t), "Y": np.cos(t)})
    return SessionData(
        meta={"year": 2024, "event": "Test"},
        drivers=drivers,
        time_axis=t,
        positions=positions,
        track_outline=outline,
    )


def test_build_track_figure_has_frames_and_cars():
    fig = build_track_figure(_fake_session_data(), highlight="VER")
    # One outline trace + one markers trace.
    assert len(fig.data) == 2
    # Animation frames exist for scrubbing/play.
    assert len(fig.frames) > 0
    # Slider for scrubbing is configured.
    assert fig.layout.sliders


def test_build_track_figure_empty_is_graceful():
    empty = SessionData()
    fig = build_track_figure(empty)
    assert fig.layout.annotations  # shows a message, does not raise


def test_friendly_error_detects_401():
    msg = _friendly_error("Could not load.", RuntimeError("HTTP Error 401"))
    assert "401" in msg
    assert "#978" in msg
    assert "2024" in msg


def test_friendly_error_detects_network():
    msg = _friendly_error("x", RuntimeError("Connection timed out"))
    assert "network" in msg.lower()


def test_timedelta_seconds_helper():
    series = pd.to_timedelta([90.5, 91.0], unit="s")
    out = analysis._timedelta_seconds(series)
    assert list(out) == [90.5, 91.0]


def test_to_int_handles_nan_and_floats():
    assert analysis._to_int(float("nan")) is None
    assert analysis._to_int(None) is None
    assert analysis._to_int(3.0) == 3
    assert analysis._to_int("5") == 5


def test_mode_or_blank():
    assert analysis._mode_or_blank(pd.Series(["SOFT", "SOFT", "MEDIUM"])) == "SOFT"
    assert analysis._mode_or_blank(pd.Series([np.nan, np.nan])) == ""
