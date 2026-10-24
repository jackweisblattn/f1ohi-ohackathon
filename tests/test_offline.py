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
from data import openf1_source as of1  # noqa: E402
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


# -- OpenF1 near-live source (all network-free, fed fake JSON-like dicts) ----


def test_openf1_parse_json_array_accepts_list():
    payload = b'[{"a": 1}, {"a": 2}]'
    out = of1._parse_json_array(payload, "sessions")
    assert out == [{"a": 1}, {"a": 2}]


def test_openf1_parse_json_array_raises_on_error_object():
    import pytest

    payload = b'{"detail": "not found"}'
    with pytest.raises(of1.SessionLoadError):
        of1._parse_json_array(payload, "sessions")


def test_openf1_parse_iso_variants():
    a = of1._parse_iso("2023-09-16T13:03:35.200000+00:00")
    b = of1._parse_iso("2023-09-16T13:03:36Z")
    assert a is not None and b is not None
    assert b > a
    assert of1._parse_iso("not-a-date") is None
    assert of1._parse_iso(None) is None


def test_openf1_seconds_from_first():
    ts = [
        "2023-09-16T13:00:00Z",
        "2023-09-16T13:00:00.500000Z",
        "2023-09-16T13:00:01Z",
    ]
    secs = of1._seconds_from_first(ts)
    assert secs[0] == 0.0
    assert abs(secs[1] - 0.5) < 1e-6
    assert abs(secs[2] - 1.0) < 1e-6


def test_openf1_build_driver_info_prefixes_colour():
    recs = [
        {
            "driver_number": 1,
            "name_acronym": "VER",
            "full_name": "Max Verstappen",
            "team_name": "Red Bull Racing",
            "team_colour": "3671C6",
        },
        {
            "driver_number": 44,
            "name_acronym": "HAM",
            "full_name": "Lewis Hamilton",
            "team_name": "Mercedes",
            "team_colour": "#27F4D2",  # already has '#'
        },
    ]
    drivers = of1.build_driver_info(recs)
    assert set(drivers) == {"VER", "HAM"}
    assert drivers["VER"].team_color == "#3671C6"
    assert drivers["HAM"].team_color == "#27F4D2"
    assert drivers["VER"].driver_number == "1"


def test_openf1_build_driver_info_handles_missing_colour():
    recs = [{"driver_number": 7, "name_acronym": "RAI", "full_name": "Kimi"}]
    drivers = of1.build_driver_info(recs)
    assert drivers["RAI"].team_color == "#888888"


def test_openf1_resample_positions_shared_axis():
    loc = {
        "VER": [
            {"date": "2023-09-16T13:00:00Z", "x": 0.0, "y": 0.0 + 10},
            {"date": "2023-09-16T13:00:01Z", "x": 10.0, "y": 10.0},
        ],
        "HAM": [
            {"date": "2023-09-16T13:00:00Z", "x": 5.0, "y": 5.0},
            {"date": "2023-09-16T13:00:02Z", "x": 25.0, "y": 25.0},
        ],
    }
    time_axis, positions = of1.resample_positions(loc, step_s=0.5)
    assert len(time_axis) > 0
    assert set(positions) == {"VER", "HAM"}
    # Both drivers resampled onto the same axis.
    assert len(positions["VER"]) == len(time_axis)
    assert len(positions["HAM"]) == len(time_axis)
    # Interpolated midpoint for HAM at t=1.0 should be ~15.
    mid = positions["HAM"]["X"].iloc[2]
    assert abs(mid - 15.0) < 1e-6


def test_openf1_resample_positions_empty():
    time_axis, positions = of1.resample_positions({}, step_s=0.5)
    assert len(time_axis) == 0
    assert positions == {}


def test_openf1_build_track_outline_picks_densest():
    loc = {
        "VER": [{"date": f"t{i}", "x": float(i), "y": float(i)} for i in range(5)],
        "HAM": [{"date": "t0", "x": 1.0, "y": 1.0}],
    }
    outline = of1.build_track_outline(loc)
    # Densest trace is VER with 5 points.
    assert len(outline) == 5
    assert list(outline.columns) == ["X", "Y"]


def test_openf1_latest_position_per_driver_takes_newest_and_skips_origin():
    records = [
        {"driver_number": 1, "date": "2023-09-16T13:00:00Z", "x": 1.0, "y": 1.0},
        {"driver_number": 1, "date": "2023-09-16T13:00:02Z", "x": 9.0, "y": 9.0},
        {"driver_number": 44, "date": "2023-09-16T13:00:01Z", "x": 0.0, "y": 0.0},
    ]
    latest = of1.latest_position_per_driver(records)
    assert latest["1"][0] == 9.0 and latest["1"][1] == 9.0
    # The (0,0) origin placeholder for 44 is dropped.
    assert "44" not in latest


def test_openf1_latest_position_standings():
    records = [
        {"driver_number": 1, "date": "2023-09-16T13:00:00Z", "position": 2},
        {"driver_number": 1, "date": "2023-09-16T13:00:05Z", "position": 1},
        {"driver_number": 44, "date": "2023-09-16T13:00:05Z", "position": 2},
    ]
    standings = of1.latest_position_standings(records)
    assert standings["1"] == 1
    assert standings["44"] == 2


def test_openf1_latest_gap_to_leader():
    records = [
        {"driver_number": 44, "date": "2023-09-16T13:00:00Z", "gap_to_leader": 1.1},
        {"driver_number": 44, "date": "2023-09-16T13:00:05Z", "gap_to_leader": 2.5},
    ]
    gaps = of1.latest_gap_to_leader(records)
    assert gaps["44"] == "2.5"


def test_openf1_to_float():
    assert of1._to_float("3.5") == 3.5
    assert of1._to_float(None) is None
    assert of1._to_float("abc") is None
    assert of1._to_float(float("nan")) is None


def test_openf1_friendly_live_error_hints():
    msg = of1._friendly_live_error("x", RuntimeError("Connection timed out"))
    assert "timed out" in msg.lower()
    msg2 = of1._friendly_live_error("x", RuntimeError("getaddrinfo failed"))
    assert "online" in msg2.lower()


def test_openf1_load_from_fakes_builds_sessiondata(monkeypatch):
    """End-to-end load() with the network stubbed by fake JSON responses."""
    src = of1.OpenF1DataSource()

    def fake_get(path, params, timeout=None):
        if path == "sessions":
            return [
                {
                    "session_key": 1234,
                    "session_name": "Race",
                    "country_name": "Singapore",
                    "year": 2023,
                    "date_start": "2023-09-17T12:00:00Z",
                }
            ]
        if path == "drivers":
            return [
                {
                    "driver_number": 1,
                    "name_acronym": "VER",
                    "full_name": "Max Verstappen",
                    "team_name": "Red Bull Racing",
                    "team_colour": "3671C6",
                },
                {
                    "driver_number": 44,
                    "name_acronym": "HAM",
                    "full_name": "Lewis Hamilton",
                    "team_name": "Mercedes",
                    "team_colour": "27F4D2",
                },
            ]
        if path == "location":
            num = params.get("driver_number")
            base = 10.0 if str(num) == "1" else 20.0
            return [
                {"date": "2023-09-17T12:00:00Z", "x": base, "y": base},
                {"date": "2023-09-17T12:00:01Z", "x": base + 5, "y": base + 5},
            ]
        return []

    monkeypatch.setattr(src, "_get", fake_get)
    data = src.load(2023, "Singapore", "R")
    assert data.raw_session is None  # OpenF1 has no FastF1 session
    assert set(data.drivers) == {"VER", "HAM"}
    assert data.drivers["VER"].team_color == "#3671C6"
    assert set(data.positions) == {"VER", "HAM"}
    assert len(data.time_axis) > 0
    assert data.meta["source"] == "openf1"
    assert data.meta["session_key"] == 1234


def test_openf1_load_live_frame_no_session(monkeypatch):
    """When /sessions?latest is empty, the live frame is not-live, not a crash."""
    src = of1.OpenF1DataSource()
    monkeypatch.setattr(src, "_get", lambda *a, **k: [])
    frame = src.load_live_frame(session_key="latest")
    assert frame.is_live is False
    assert frame.positions == {}


def test_openf1_load_live_frame_with_positions(monkeypatch):
    src = of1.OpenF1DataSource()

    def fake_get(path, params, timeout=None):
        if path == "sessions":
            return [{"session_key": 99, "session_name": "Race"}]
        if path == "drivers":
            return [
                {
                    "driver_number": 1,
                    "name_acronym": "VER",
                    "full_name": "Max",
                    "team_name": "RBR",
                    "team_colour": "3671C6",
                }
            ]
        if path == "location":
            return [
                {"driver_number": 1, "date": "2023-09-17T12:00:05Z", "x": 7.0, "y": 8.0}
            ]
        if path == "position":
            return [{"driver_number": 1, "date": "2023-09-17T12:00:05Z", "position": 1}]
        if path == "intervals":
            return []
        return []

    monkeypatch.setattr(src, "_get", fake_get)
    frame = src.load_live_frame(session_key="latest")
    assert frame.is_live is True
    assert frame.positions["VER"] == (7.0, 8.0)
    assert frame.session_name == "Race"
    assert frame.standings and frame.standings[0]["abbreviation"] == "VER"
