"""Animated 2D track-map rendering with Plotly.

This module turns a loaded :class:`~data.loader.SessionData` into an animated
Plotly figure: the circuit outline drawn as a line, plus one coloured dot per
driver (official team colour) labelled with the 3-letter abbreviation. Plotly's
built-in animation frames drive the play/pause control and the time slider acts
as the scrubber.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _axis_range(track_outline: pd.DataFrame, positions: dict[str, pd.DataFrame]):
    """Compute padded, equal-aspect axis ranges covering track + cars."""
    xs: list[float] = []
    ys: list[float] = []
    if not track_outline.empty:
        xs += track_outline["X"].dropna().tolist()
        ys += track_outline["Y"].dropna().tolist()
    for frame in positions.values():
        xs += frame["X"].dropna().tolist()
        ys += frame["Y"].dropna().tolist()
    if not xs or not ys:
        return None, None
    pad_x = (max(xs) - min(xs)) * 0.05 or 1.0
    pad_y = (max(ys) - min(ys)) * 0.05 or 1.0
    return (min(xs) - pad_x, max(xs) + pad_x), (min(ys) - pad_y, max(ys) + pad_y)


def _frame_markers(
    session_data,
    highlight: Optional[str],
    frame_index: int,
) -> go.Scatter:
    """Build the per-frame marker scatter (one point per driver with a label)."""
    xs, ys, colors, texts, sizes, lines = [], [], [], [], [], []
    for abbr, frame in session_data.positions.items():
        info = session_data.drivers.get(abbr)
        color = info.team_color if info else "#888888"
        x = frame["X"].iloc[frame_index]
        y = frame["Y"].iloc[frame_index]
        xs.append(x)
        ys.append(y)
        colors.append(color)
        texts.append(abbr)
        if highlight and abbr == highlight:
            sizes.append(20)
            lines.append(2.5)
        else:
            sizes.append(13)
            lines.append(0.5)

    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=texts,
        textposition="top center",
        textfont=dict(size=10, color="#ffffff"),
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=lines, color="#ffffff"),
        ),
        hovertext=texts,
        hoverinfo="text",
        name="cars",
        showlegend=False,
    )


def build_track_figure(
    session_data,
    highlight: Optional[str] = None,
    max_frames: int = 600,
) -> go.Figure:
    """Build an animated Plotly figure for the whole session.

    Parameters
    ----------
    session_data:
        A loaded :class:`~data.loader.SessionData`.
    highlight:
        Optional driver abbreviation to emphasise (larger, outlined dot).
    max_frames:
        Hard cap on animation frames to keep the browser responsive. The common
        time axis is strided down to at most this many frames.
    """
    time_axis = session_data.time_axis
    if time_axis is None or len(time_axis) == 0 or not session_data.positions:
        return _empty_figure("No position data available for this session.")

    # Stride the time axis down to <= max_frames so ~20 animated cars stay smooth.
    n = len(time_axis)
    stride = max(1, int(np.ceil(n / max_frames)))
    frame_indices = list(range(0, n, stride))

    xr, yr = _axis_range(session_data.track_outline, session_data.positions)

    fig = go.Figure()

    # Static circuit outline underneath the cars.
    if not session_data.track_outline.empty:
        fig.add_trace(
            go.Scatter(
                x=session_data.track_outline["X"],
                y=session_data.track_outline["Y"],
                mode="lines",
                line=dict(color="#555555", width=3),
                name="circuit",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Initial car positions (first frame).
    fig.add_trace(_frame_markers(session_data, highlight, frame_indices[0]))

    # Animation frames + slider steps.
    frames = []
    slider_steps = []
    base_trace_count = len(fig.data) - 1  # index of the markers trace
    for fi in frame_indices:
        t = float(time_axis[fi])
        frames.append(
            go.Frame(
                name=f"{t:.1f}",
                data=[_frame_markers(session_data, highlight, fi)],
                traces=[base_trace_count],
            )
        )
        slider_steps.append(
            dict(
                method="animate",
                label=_fmt_mmss(t),
                args=[
                    [f"{t:.1f}"],
                    dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                    ),
                ],
            )
        )
    fig.frames = frames

    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(visible=False, range=xr),
        yaxis=dict(visible=False, range=yr, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=30, b=10),
        height=620,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=0.02,
                xanchor="left",
                yanchor="bottom",
                direction="left",
                buttons=[
                    dict(
                        label="\u25b6 Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=60, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="\u23f8 Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.1,
                y=0,
                len=0.9,
                xanchor="left",
                yanchor="top",
                currentvalue=dict(prefix="Time: ", visible=True),
                pad=dict(t=30, b=10),
                steps=slider_steps,
            )
        ],
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                showarrow=False,
                font=dict(size=16, color="#cccccc"),
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
            )
        ],
    )
    return fig


def _fmt_mmss(seconds: float) -> str:
    """Format seconds-from-start as ``M:SS``."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"
