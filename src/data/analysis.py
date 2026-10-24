"""Per-driver analysis built on top of a loaded :class:`SessionData`.

Everything here operates on the underlying FastF1 ``Session`` carried by
``SessionData.raw_session`` plus the driver metadata. The functions are written
defensively so that a driver with missing/partial data yields an empty result
rather than raising, which keeps the Streamlit UI robust.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class StintDegradation:
    """Tire-degradation estimate for a single stint.

    ``deg_s_per_lap`` is the slope of a linear fit of lap time (seconds) against
    lap number within the stint: a positive value means the car got slower as
    the tires aged.
    """

    stint: int
    compound: str
    n_laps: int
    start_lap: int
    end_lap: int
    avg_laptime_s: float
    deg_s_per_lap: float


@dataclass
class DriverSummary:
    """Headline stats for a driver, shown in the stats panel."""

    abbreviation: str
    full_name: str
    team_name: str
    grid_position: Optional[int]
    current_position: Optional[int]
    position_change: Optional[int]  # grid - current (positive = gained places)
    current_compound: Optional[str]
    current_tyre_age: Optional[int]
    best_laptime_s: Optional[float]
    last_laptime_s: Optional[float]


def _driver_laps(session, abbreviation: str) -> pd.DataFrame:
    """Return the laps DataFrame for a driver, or an empty frame."""
    try:
        laps = session.laps.pick_drivers(abbreviation)
    except Exception:  # noqa: BLE001 - older FastF1 used pick_driver
        try:
            laps = session.laps.pick_driver(abbreviation)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    return laps.reset_index(drop=True) if laps is not None else pd.DataFrame()


def lap_times(session, abbreviation: str) -> pd.DataFrame:
    """Return a tidy per-lap table for a driver.

    Columns: ``LapNumber``, ``LapTime`` (seconds), ``Compound``, ``TyreLife``,
    ``Stint``, ``Position``. Laps without a valid time are dropped.
    """
    laps = _driver_laps(session, abbreviation)
    if laps.empty:
        return pd.DataFrame(
            columns=[
                "LapNumber",
                "LapTime",
                "Compound",
                "TyreLife",
                "Stint",
                "Position",
            ]
        )

    out = pd.DataFrame()
    out["LapNumber"] = _col(laps, "LapNumber")
    out["LapTime"] = _timedelta_seconds(_col(laps, "LapTime"))
    out["Compound"] = _col(laps, "Compound")
    out["TyreLife"] = _col(laps, "TyreLife")
    out["Stint"] = _col(laps, "Stint")
    out["Position"] = _col(laps, "Position")
    out = out.dropna(subset=["LapTime"]).reset_index(drop=True)
    return out


def tire_degradation(session, abbreviation: str) -> list[StintDegradation]:
    """Estimate tire degradation per stint via a linear lap-time trend.

    For each stint we fit ``laptime ~ lapnumber`` and report the slope in
    seconds-per-lap. Outlier laps (in/out laps, safety cars) are reduced in
    influence by excluding laps slower than the stint median by > 5s.
    """
    laps = lap_times(session, abbreviation)
    if laps.empty or "Stint" not in laps:
        return []

    results: list[StintDegradation] = []
    for stint, grp in laps.groupby("Stint", dropna=True):
        grp = grp.dropna(subset=["LapTime", "LapNumber"])
        if len(grp) < 2:
            continue
        median = grp["LapTime"].median()
        clean = grp[grp["LapTime"] <= median + 5.0]
        if len(clean) < 2:
            clean = grp

        x = clean["LapNumber"].to_numpy(dtype=float)
        y = clean["LapTime"].to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else float("nan")

        compound = _mode_or_blank(grp["Compound"])
        results.append(
            StintDegradation(
                stint=int(stint),
                compound=compound,
                n_laps=int(len(grp)),
                start_lap=int(grp["LapNumber"].min()),
                end_lap=int(grp["LapNumber"].max()),
                avg_laptime_s=float(grp["LapTime"].mean()),
                deg_s_per_lap=slope,
            )
        )
    results.sort(key=lambda r: r.stint)
    return results


def driver_summary(session, drivers, abbreviation: str) -> DriverSummary:
    """Build the headline stats for a driver's panel."""
    info = drivers.get(abbreviation)
    full_name = info.full_name if info else abbreviation
    team_name = info.team_name if info else ""

    laps = lap_times(session, abbreviation)

    grid = _result_value(session, abbreviation, "GridPosition")
    current = _result_value(session, abbreviation, "Position")
    if current is None and not laps.empty and laps["Position"].notna().any():
        current = laps["Position"].dropna().iloc[-1]

    grid_i = _to_int(grid)
    current_i = _to_int(current)
    change = (
        grid_i - current_i if grid_i is not None and current_i is not None else None
    )

    compound = None
    tyre_age = None
    best = None
    last = None
    if not laps.empty:
        last_row = laps.iloc[-1]
        compound = _none_if_nan(last_row.get("Compound"))
        tyre_age = _to_int(last_row.get("TyreLife"))
        best = float(laps["LapTime"].min())
        last = float(last_row["LapTime"])

    return DriverSummary(
        abbreviation=abbreviation,
        full_name=full_name,
        team_name=team_name,
        grid_position=grid_i,
        current_position=current_i,
        position_change=change,
        current_compound=compound,
        current_tyre_age=tyre_age,
        best_laptime_s=best,
        last_laptime_s=last,
    )


def driver_telemetry(session, abbreviation: str, lap: str = "fastest") -> pd.DataFrame:
    """Return a telemetry trace (speed/throttle/brake/gear/DRS) for one lap.

    ``lap`` may be ``"fastest"`` (default) or a lap number. Used by the stretch
    telemetry feature. Returns an empty frame if telemetry is unavailable.
    """
    laps = _driver_laps(session, abbreviation)
    if laps.empty:
        return pd.DataFrame()
    try:
        if lap == "fastest":
            chosen = laps.pick_fastest()
        else:
            chosen = laps[laps["LapNumber"] == float(lap)].iloc[0]
        tel = chosen.get_telemetry()
    except Exception:  # noqa: BLE001 - telemetry missing for this lap
        return pd.DataFrame()

    wanted = ["Distance", "Speed", "Throttle", "Brake", "nGear", "DRS"]
    keep = [c for c in wanted if c in tel]
    out = tel.loc[:, keep].reset_index(drop=True)
    return out


# -- small, defensive helpers ------------------------------------------------


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([np.nan] * len(df))


def _timedelta_seconds(series: pd.Series) -> pd.Series:
    """Convert a timedelta (or already-numeric) series to float seconds."""
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return pd.to_numeric(series, errors="coerce")


def _mode_or_blank(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return ""
    try:
        return str(s.mode().iloc[0])
    except Exception:  # noqa: BLE001
        return str(s.iloc[0])


def _result_value(session, abbreviation: str, column: str):
    try:
        results = session.results
        row = results[results["Abbreviation"] == abbreviation]
        if row.empty or column not in row:
            return None
        return row.iloc[0][column]
    except Exception:  # noqa: BLE001 - results may be unavailable
        return None


def _to_int(value) -> Optional[int]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _none_if_nan(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value
