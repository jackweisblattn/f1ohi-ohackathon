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
)
from viz.track_map import build_track_figure  # noqa: E402

st.set_page_config(page_title="F1 Pit Wall", page_icon="\U0001f3ce", layout="wide")


@st.cache_resource(show_spinner=False)
def get_source():
    """Return the (cached) data source instance."""
    return default_source()


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
    """Render the per-driver stats panel (core features 2 & 3)."""
    raw = session_data.raw_session
    info = session_data.drivers.get(abbr)
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


def main():
    st.title("\U0001f3ce\ufe0f F1 Pit Wall")
    st.caption(
        "Replay a completed F1 session on an animated track map. "
        "Turning raw public F1 data into an intuitive visual tool."
    )

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


if __name__ == "__main__":
    main()
