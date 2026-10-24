"""F1 Pit Wall - Streamlit dashboard entry point.

Run with::

    streamlit run src/app.py

The app replays a completed Formula 1 session on an animated 2D track map. Each
driver is a coloured dot (official team colour + 3-letter abbreviation) moving
around the circuit, with play/pause/scrub controls. Selecting a driver opens a
stats panel with lap times, position change, tire info and a per-stint tire
degradation estimate. An optional telemetry tab shows speed/throttle/brake/gear/
DRS for the selected driver's fastest lap and a two-driver overlay.

The data layer is accessed only through the abstract
:class:`~data.loader.SessionDataSource`, so the FastF1 historical source can be
swapped for an OpenF1 near-live source later without touching this file.
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make ``src`` importable whether run via ``streamlit run src/app.py`` or as a
# module, so the intra-package imports below resolve consistently.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import analysis  # noqa: E402
from data.loader import (  # noqa: E402
    DEFAULT_EVENT,
    DEFAULT_SESSION,
    DEFAULT_YEAR,
    SESSION_TYPES,
    SessionLoadError,
    default_source,
    live_source,
)
from viz.track_map import build_track_figure  # noqa: E402

st.set_page_config(page_title="F1 Pit Wall", page_icon="\U0001f3ce", layout="wide")

# How often (seconds) Live mode polls OpenF1 for new location samples.
LIVE_REFRESH_S = 3


@st.cache_resource(show_spinner=False)
def get_source():
    """Return the (cached) FastF1 replay data source instance."""
    return default_source()


@st.cache_resource(show_spinner=False)
def get_live_source():
    """Return the (cached) OpenF1 near-live data source instance."""
    return live_source()


@st.cache_data(show_spinner=True, ttl=3600)
def load_session_cached(year: int, event: str, session_type: str):
    """Load a session, cached by (year, event, session_type).

    Returns a ``(SessionData | None, error_message | None)`` tuple so Streamlit
    can show a clean message instead of a traceback.
    """
    source = get_source()
    try:
        return source.load(year, event, session_type), None
    except SessionLoadError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - last-resort guard for the UI
        return None, f"Unexpected error while loading the session:\n\n{exc}"


@st.cache_data(show_spinner=False, ttl=3600)
def event_list_cached(year: int) -> list[str]:
    """Return event names for a year, or an empty list on failure."""
    try:
        return get_source().list_events(year)
    except SessionLoadError:
        return []


def fmt_time(seconds) -> str:
    """Format a lap time in seconds as ``M:SS.mmm``."""
    if seconds is None or pd.isna(seconds):
        return "-"
    m, s = divmod(float(seconds), 60)
    return f"{int(m)}:{s:06.3f}"


def sidebar_session_picker():
    """Render the session selector and return the chosen (year, event, type)."""
    st.sidebar.header("Session")

    source = get_source()
    years = source.list_years()
    default_year_idx = years.index(DEFAULT_YEAR) if DEFAULT_YEAR in years else 0
    year = st.sidebar.selectbox("Year", years, index=default_year_idx)

    events = event_list_cached(year)
    if events:
        default_event_idx = (
            events.index(DEFAULT_EVENT) if DEFAULT_EVENT in events else 0
        )
        event = st.sidebar.selectbox("Event", events, index=default_event_idx)
    else:
        st.sidebar.info(
            "Could not fetch the event schedule (offline or blocked). "
            "Type an event name manually."
        )
        event = st.sidebar.text_input("Event", value=DEFAULT_EVENT)

    type_labels = [label for _, label in SESSION_TYPES]
    type_ids = [sid for sid, _ in SESSION_TYPES]
    default_type_idx = (
        type_ids.index(DEFAULT_SESSION) if DEFAULT_SESSION in type_ids else 0
    )
    type_label = st.sidebar.selectbox("Session", type_labels, index=default_type_idx)
    session_type = type_ids[type_labels.index(type_label)]

    load_clicked = st.sidebar.button("Load session", type="primary")
    return year, event, session_type, load_clicked


def render_stats_panel(session_data, abbr: str):
    """Render the per-driver stats panel (core features 2 & 3).

    The full lap/tire/telemetry analysis relies on a FastF1 ``raw_session``.
    OpenF1-backed sessions have ``raw_session is None``; in that case we degrade
    gracefully to just the driver identity rather than crashing.
    """
    raw = session_data.raw_session
    info = session_data.drivers.get(abbr)

    if raw is None:
        if info is not None:
            st.markdown(
                f"### <span style='color:{info.team_color}'>\u25cf</span> "
                f"{info.abbreviation} &mdash; {info.full_name}",
                unsafe_allow_html=True,
            )
            st.caption(info.team_name)
        st.info(
            "Detailed lap, tire and telemetry stats are only available in "
            "**Replay (FastF1)** mode. In **Live (OpenF1)** mode the track map "
            "shows near-live car positions, plus any available standings/gaps."
        )
        return

    summary = analysis.driver_summary(raw, session_data.drivers, abbr)

    st.markdown(
        f"### <span style='color:{info.team_color}'>\u25cf</span> "
        f"{summary.abbreviation} &mdash; {summary.full_name}",
        unsafe_allow_html=True,
    )
    st.caption(summary.team_name)

    c1, c2, c3 = st.columns(3)
    c1.metric("Current position", summary.current_position or "-")
    delta = (
        f"{summary.position_change:+d}" if summary.position_change is not None else None
    )
    c2.metric("Grid", summary.grid_position or "-", delta=delta)
    c3.metric(
        "Tire",
        summary.current_compound or "-",
        delta=(
            f"{summary.current_tyre_age} laps"
            if summary.current_tyre_age is not None
            else None
        ),
        delta_color="off",
    )

    c4, c5 = st.columns(2)
    c4.metric("Best lap", fmt_time(summary.best_laptime_s))
    c5.metric("Last lap", fmt_time(summary.last_laptime_s))

    laps = analysis.lap_times(raw, abbr)
    if not laps.empty:
        st.markdown("#### Lap times")
        chart = laps.set_index("LapNumber")["LapTime"]
        st.line_chart(chart, height=200)

    deg = analysis.tire_degradation(raw, abbr)
    if deg:
        st.markdown("#### Tire degradation by stint")
        deg_df = pd.DataFrame(
            [
                {
                    "Stint": d.stint,
                    "Compound": d.compound,
                    "Laps": d.n_laps,
                    "Avg lap (s)": round(d.avg_laptime_s, 3),
                    "Deg (s/lap)": round(d.deg_s_per_lap, 3),
                }
                for d in deg
            ]
        )
        st.dataframe(deg_df, hide_index=True, use_container_width=True)
        st.caption(
            "Deg (s/lap) is the slope of lap time vs lap number within a stint. "
            "Positive means the tires slowed the car as they aged."
        )


def render_telemetry_tab(session_data, primary: str):
    """Stretch feature: telemetry traces + optional two-driver overlay."""
    raw = session_data.raw_session
    others = ["(none)"] + [a for a in sorted(session_data.positions) if a != primary]
    compare = st.selectbox("Compare with", others, index=0)

    tel = analysis.driver_telemetry(raw, primary, lap="fastest")
    if tel.empty or "Distance" not in tel:
        st.info("No telemetry available for this driver's fastest lap.")
        return

    channels = [c for c in ["Speed", "Throttle", "Brake", "nGear", "DRS"] if c in tel]
    primary_color = session_data.drivers[primary].team_color

    tel2 = pd.DataFrame()
    compare_color = "#ff4b4b"
    if compare != "(none)":
        tel2 = analysis.driver_telemetry(raw, compare, lap="fastest")
        compare_color = session_data.drivers[compare].team_color

    for ch in channels:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=tel["Distance"],
                y=tel[ch],
                mode="lines",
                name=primary,
                line=dict(color=primary_color),
            )
        )
        if not tel2.empty and ch in tel2:
            fig.add_trace(
                go.Scatter(
                    x=tel2["Distance"],
                    y=tel2[ch],
                    mode="lines",
                    name=compare,
                    line=dict(color=compare_color),
                )
            )
        fig.update_layout(
            template="plotly_dark",
            height=220,
            margin=dict(l=10, r=10, t=30, b=10),
            title=ch,
            xaxis_title="Distance (m)",
        )
        st.plotly_chart(fig, use_container_width=True)


def _autorefresh(interval_ms: int, key: str) -> None:
    """Trigger a periodic rerun for Live mode, across Streamlit versions.

    Prefers the built-in ``st.autorefresh`` (newer Streamlit). Falls back to a
    ``time.sleep`` + ``st.rerun`` loop so no extra dependency is required.
    """
    auto = getattr(st, "autorefresh", None)
    if callable(auto):
        auto(interval=interval_ms, key=key)
        return
    # Fallback: sleep then rerun. This blocks the script briefly but needs no
    # third-party package. Kept short so the UI stays responsive.
    time.sleep(max(0.5, interval_ms / 1000.0))
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()


def render_live_mode():
    """Render the Live (OpenF1) near-live view.

    Polls OpenF1 for the most recent car positions every few seconds and draws
    them on a static circuit outline. When no session is running on track it
    shows a clear message instead of a blank screen.
    """
    st.sidebar.header("Live session")
    st.sidebar.caption(
        "Targets the current or most recent session via OpenF1 "
        "(`session_key=latest`)."
    )
    paused = st.sidebar.checkbox("Pause auto-refresh", value=False)

    source = get_live_source()

    try:
        frame = source.load_live_frame(session_key="latest")
    except SessionLoadError as exc:
        st.error(str(exc))
        st.stop()
        return
    except Exception as exc:  # noqa: BLE001 - last-resort guard for the UI
        st.error(f"Unexpected error while polling OpenF1:\n\n{exc}")
        st.stop()
        return

    label = frame.session_name or "latest session"
    st.subheader(f"\U0001f534 Live \u2014 {label}")

    if not frame.is_live or not frame.positions:
        st.warning(
            "**No live session is running on track right now.**\n\n"
            "OpenF1 only streams car positions while a session is actually "
            "*green* (cars on track, ~3s delay). There is nothing to animate "
            "until then.\n\n"
            "For a demo when no session is live, switch **Mode** to "
            "**Replay (FastF1)** in the sidebar and play back a completed race. "
            "This Live view will populate automatically once a session goes "
            "green."
        )
        # Keep polling so it lights up the moment a session starts.
        if not paused:
            _autorefresh(LIVE_REFRESH_S * 1000, key="live_refresh_idle")
        return

    # Build a lightweight SessionData-like view for the existing figure.
    live_session = _live_session_data(frame)
    drivers = sorted(live_session.positions.keys())
    selected = st.sidebar.selectbox("Driver", drivers, index=0) if drivers else None

    map_col, panel_col = st.columns([3, 2], gap="large")
    with map_col:
        fig = build_track_figure(live_session, highlight=selected)
        st.plotly_chart(fig, use_container_width=True)
        if frame.latest_ts:
            st.caption(f"Latest sample: {frame.latest_ts} (UTC) \u00b7 ~3s behind live")

    with panel_col:
        if selected:
            render_stats_panel(live_session, selected)
        if frame.standings:
            st.markdown("#### Live standings")
            st.dataframe(
                pd.DataFrame(frame.standings),
                hide_index=True,
                use_container_width=True,
            )

    if not paused:
        _autorefresh(LIVE_REFRESH_S * 1000, key="live_refresh")


def _live_session_data(frame):
    """Adapt a :class:`LiveFrame` to a minimal ``SessionData`` for the figure.

    The track map only needs ``positions`` (DataFrames with X/Y), ``drivers``
    and a (possibly empty) ``track_outline``/``time_axis``. We give each driver
    a single-row frame at the current position; the circuit outline accumulates
    across refreshes so the track shape fills in as cars move.
    """
    import numpy as np

    from data.loader import SessionData

    # Accumulate an outline from observed positions across refreshes.
    trail = st.session_state.setdefault("live_outline", [])
    for x, y in frame.positions.values():
        if x is not None and y is not None:
            trail.append((x, y))
    # Bound memory: keep the most recent points.
    if len(trail) > 5000:
        del trail[: len(trail) - 5000]

    positions = {}
    for abbr, (x, y) in frame.positions.items():
        positions[abbr] = pd.DataFrame({"X": [x], "Y": [y]}, index=[0.0])

    outline = (
        pd.DataFrame(trail, columns=["X", "Y"])
        if trail
        else pd.DataFrame(columns=["X", "Y"])
    )

    return SessionData(
        meta={"source": "openf1", "session_name": frame.session_name},
        drivers=frame.drivers,
        time_axis=np.array([0.0]),
        positions=positions,
        track_outline=outline,
        raw_session=None,
    )


def render_replay_mode():
    """Render the Replay (FastF1) view (the original dashboard behaviour)."""
    year, event, session_type, load_clicked = sidebar_session_picker()

    # Load on first run (defaults) or when the button is pressed.
    key = (year, event, session_type)
    if load_clicked or "session_key" not in st.session_state:
        session_data, error = load_session_cached(*key)
        st.session_state["session_key"] = key
        st.session_state["session_data"] = session_data
        st.session_state["session_error"] = error

    session_data = st.session_state.get("session_data")
    error = st.session_state.get("session_error")

    if error:
        st.error(error)
        st.stop()
    if session_data is None:
        st.info("Pick a session in the sidebar and click **Load session**.")
        st.stop()

    meta = session_data.meta
    st.subheader(
        f"{meta.get('year')} {meta.get('event')} \u2014 "
        f"{meta.get('session_name', meta.get('session_type'))}"
    )

    drivers = sorted(session_data.positions.keys())
    if not drivers:
        st.warning("This session has no position data to animate.")
        st.stop()

    # Dropdown is the reliable primary selection method (Plotly click-to-select
    # inside Streamlit is unreliable), per the design guidance.
    selected = st.sidebar.selectbox("Driver", drivers, index=0)

    map_col, panel_col = st.columns([3, 2], gap="large")
    with map_col:
        fig = build_track_figure(session_data, highlight=selected)
        st.plotly_chart(fig, use_container_width=True)

    with panel_col:
        tab_stats, tab_tel = st.tabs(["Stats", "Telemetry"])
        with tab_stats:
            render_stats_panel(session_data, selected)
        with tab_tel:
            render_telemetry_tab(session_data, selected)


def main():
    st.title("\U0001f3ce\ufe0f F1 Pit Wall")
    st.caption(
        "Animated F1 track map. Replay a completed session (FastF1) or follow a "
        "near-live session (OpenF1, ~3s behind)."
    )

    mode = st.sidebar.radio(
        "Mode",
        ["Replay (FastF1)", "Live (OpenF1)"],
        index=0,
        help=(
            "Replay plays back a completed session with full stats and "
            "telemetry. Live shows near-real-time car positions from OpenF1 "
            "while a session is running on track."
        ),
    )
    st.sidebar.divider()

    if mode == "Live (OpenF1)":
        render_live_mode()
    else:
        render_replay_mode()


if __name__ == "__main__":
    main()
