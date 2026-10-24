"""Data loading layer for the F1 Pit Wall dashboard.

The loader is deliberately hidden behind an abstract interface
(:class:`SessionDataSource`) so that the FastF1 (historical replay) source can
later be swapped for an OpenF1 (near-live) source without touching the UI or
the visualisation code. The UI only ever talks to the abstract interface and
the plain dataclasses returned from it.

Design notes
------------
* FastF1 caching is enabled against a local ``./cache`` directory. The first
  load of a session downloads 50-100MB; subsequent runs are fast and resilient.
* Per-driver position data in FastF1 is time-indexed and irregularly sampled.
  We resample/interpolate every car onto a *common* time axis (default 0.5s
  step) so the animation stays in sync and stays smooth for ~20 cars.
* Session loading is wrapped in graceful error handling. A known issue
  (FastF1 GitHub #978) is that F1's live-timing server can return HTTP 401/403
  to unauthenticated requests on a cold cache, which breaks ``Session.load()``
  for recent (2025/2026) sessions. We surface a clear, user-facing message
  instead of a raw stack trace.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Default cache directory lives at the project root (one level above ``src``).
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "cache",
)

# Sensible, reliable demo default. 2024 sessions are used because recent
# (2025/2026) sessions are more likely to hit the live-timing 401/403 issue on
# a cold cache. COTA 2024 is a well known, data-complete race.
DEFAULT_YEAR = 2024
DEFAULT_EVENT = "United States"
DEFAULT_SESSION = "R"  # Race

# Common time step (seconds) used to resample every car onto a shared axis.
DEFAULT_RESAMPLE_STEP_S = 0.5


class SessionLoadError(RuntimeError):
    """Raised when a session cannot be loaded.

    The message is intended to be shown directly to the user, so it should be
    human readable and actionable rather than a raw exception string.
    """


@dataclass
class DriverInfo:
    """Static, per-driver metadata for the current session."""

    abbreviation: str  # e.g. "VER"
    driver_number: str  # e.g. "1"
    full_name: str  # e.g. "Max Verstappen"
    team_name: str  # e.g. "Red Bull Racing"
    team_color: str  # hex colour, e.g. "#3671C6"


@dataclass
class SessionData:
    """A fully-loaded session, ready for visualisation.

    Attributes
    ----------
    meta:
        Free-form metadata describing the session (year, event, type, etc.).
    drivers:
        Mapping of driver abbreviation -> :class:`DriverInfo`.
    time_axis:
        The common, monotonically increasing time axis (seconds from session
        start) that every car's position is sampled onto.
    positions:
        Mapping of driver abbreviation -> DataFrame indexed by the common time
        axis with ``X`` and ``Y`` columns (track coordinates). Cars that are not
        on track at a given time have NaN entries.
    track_outline:
        DataFrame with ``X``/``Y`` columns describing the circuit outline,
        derived from a representative fast lap.
    raw_session:
        The underlying FastF1 ``Session`` object (or ``None`` for other
        sources). Used by the analysis layer for lap/telemetry details.
    """

    meta: dict = field(default_factory=dict)
    drivers: dict[str, DriverInfo] = field(default_factory=dict)
    time_axis: np.ndarray = field(default_factory=lambda: np.array([]))
    positions: dict[str, pd.DataFrame] = field(default_factory=dict)
    track_outline: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_session: object | None = None


class SessionDataSource(abc.ABC):
    """Abstract data source.

    Implementations provide a catalogue of selectable sessions and the ability
    to load one into a :class:`SessionData`. Keeping this abstract lets us add
    an OpenF1 (near-live) source later without changing the UI.
    """

    @abc.abstractmethod
    def list_years(self) -> list[int]:
        """Return the years that can be selected."""

    @abc.abstractmethod
    def list_events(self, year: int) -> list[str]:
        """Return the event names available for ``year``."""

    @abc.abstractmethod
    def list_session_types(self) -> list[tuple[str, str]]:
        """Return ``(identifier, label)`` pairs for selectable session types."""

    @abc.abstractmethod
    def load(
        self,
        year: int,
        event: str,
        session_type: str,
        resample_step_s: float = DEFAULT_RESAMPLE_STEP_S,
    ) -> SessionData:
        """Load a session and return a :class:`SessionData`.

        Raises
        ------
        SessionLoadError
            If the session cannot be loaded. The message is user-facing.
        """


# Common F1 session identifiers. Kept as a module constant so both the FastF1
# source and the UI share one source of truth.
SESSION_TYPES: list[tuple[str, str]] = [
    ("FP1", "Practice 1"),
    ("FP2", "Practice 2"),
    ("FP3", "Practice 3"),
    ("Q", "Qualifying"),
    ("S", "Sprint"),
    ("SQ", "Sprint Qualifying"),
    ("R", "Race"),
]


class FastF1DataSource(SessionDataSource):
    """Historical-replay data source backed by `FastF1 <https://docs.fastf1.dev>`_.

    FastF1 is imported lazily inside methods so that importing this module (for
    tests, linting, or the analysis layer) does not require FastF1 or network
    access to be present.
    """

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self._cache_enabled = False

    # -- internal helpers ------------------------------------------------

    def _ensure_cache(self) -> None:
        """Enable FastF1's on-disk cache, creating the directory if needed."""
        if self._cache_enabled:
            return
        import fastf1  # local import: optional/network-free module import

        os.makedirs(self.cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(self.cache_dir)
        self._cache_enabled = True

    # -- catalogue ------------------------------------------------------

    def list_years(self) -> list[int]:
        # FastF1 has reliable data from 2018 onwards (telemetry/position).
        current = 2024
        try:
            from datetime import datetime

            current = max(current, datetime.now().year)
        except Exception:  # pragma: no cover - defensive only
            pass
        return list(range(current, 2017, -1))

    def list_events(self, year: int) -> list[str]:
        """Return event names for ``year`` using the FastF1 event schedule."""
        try:
            self._ensure_cache()
            import fastf1

            schedule = fastf1.get_event_schedule(year, include_testing=False)
            names = [str(n) for n in schedule["EventName"].tolist()]
            return names
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            raise SessionLoadError(
                _friendly_error(f"Could not fetch the {year} event schedule.", exc)
            ) from exc

    def list_session_types(self) -> list[tuple[str, str]]:
        return list(SESSION_TYPES)

    # -- loading --------------------------------------------------------

    def load(
        self,
        year: int,
        event: str,
        session_type: str,
        resample_step_s: float = DEFAULT_RESAMPLE_STEP_S,
    ) -> SessionData:
        self._ensure_cache()
        import fastf1

        try:
            session = fastf1.get_session(year, event, session_type)
            # telemetry=True is required for X/Y position + car telemetry.
            session.load(laps=True, telemetry=True, weather=False, messages=False)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            raise SessionLoadError(
                _friendly_error(f"Could not load {year} {event} ({session_type}).", exc)
            ) from exc

        drivers = self._build_driver_info(session)
        time_axis, positions = self._build_positions(session, drivers, resample_step_s)
        track_outline = self._build_track_outline(session)

        meta = {
            "year": year,
            "event": event,
            "session_type": session_type,
            "session_name": getattr(session, "name", session_type),
            "event_name": getattr(getattr(session, "event", None), "EventName", event),
            "n_drivers": len(drivers),
            "n_frames": int(len(time_axis)),
            "resample_step_s": resample_step_s,
        }

        return SessionData(
            meta=meta,
            drivers=drivers,
            time_axis=time_axis,
            positions=positions,
            track_outline=track_outline,
            raw_session=session,
        )

    # -- builders -------------------------------------------------------

    @staticmethod
    def _build_driver_info(session) -> dict[str, DriverInfo]:
        """Build per-driver metadata (colour, abbreviation, team)."""
        import fastf1.plotting as f1plotting

        drivers: dict[str, DriverInfo] = {}
        for drv in session.drivers:
            info = session.get_driver(drv)
            abbr = str(info.get("Abbreviation", drv))
            team = str(info.get("TeamName", ""))
            color = _safe_team_color(f1plotting, team, session)
            drivers[abbr] = DriverInfo(
                abbreviation=abbr,
                driver_number=str(drv),
                full_name=str(info.get("FullName", abbr)),
                team_name=team,
                team_color=color,
            )
        return drivers

    @staticmethod
    def _build_positions(
        session, drivers: dict[str, DriverInfo], step_s: float
    ) -> tuple[np.ndarray, dict[str, pd.DataFrame]]:
        """Resample every car's X/Y position onto a shared time axis.

        Position data per car is time-indexed and irregularly sampled, so we
        reindex each car onto a common, evenly-spaced axis (in seconds from the
        session start) and linearly interpolate X/Y within each car's on-track
        span. This keeps all cars in sync and bounds the frame count.
        """
        per_driver_seconds: dict[str, pd.DataFrame] = {}
        global_max = 0.0

        for abbr, info in drivers.items():
            try:
                pos = session.pos_data[info.driver_number]
            except Exception:  # noqa: BLE001 - car may have no position data
                continue
            if pos is None or len(pos) == 0:
                continue

            frame = pos.loc[:, ["X", "Y"]].copy()
            # ``Time`` is a timedelta from session start; convert to seconds.
            seconds = pos["Time"].dt.total_seconds().to_numpy()
            frame.index = pd.Index(seconds, name="t")
            # Drop duplicate/non-monotonic timestamps that break interpolation.
            frame = frame[~frame.index.duplicated(keep="first")].sort_index()
            if frame.empty:
                continue
            per_driver_seconds[abbr] = frame
            global_max = max(global_max, float(frame.index.max()))

        if global_max <= 0 or not per_driver_seconds:
            return np.array([]), {}

        time_axis = np.arange(0.0, global_max + step_s, step_s)

        positions: dict[str, pd.DataFrame] = {}
        for abbr, frame in per_driver_seconds.items():
            xs = np.interp(
                time_axis,
                frame.index.to_numpy(),
                frame["X"].to_numpy(),
                left=np.nan,
                right=np.nan,
            )
            ys = np.interp(
                time_axis,
                frame.index.to_numpy(),
                frame["Y"].to_numpy(),
                left=np.nan,
                right=np.nan,
            )
            positions[abbr] = pd.DataFrame(
                {"X": xs, "Y": ys}, index=pd.Index(time_axis, name="t")
            )

        return time_axis, positions

    @staticmethod
    def _build_track_outline(session) -> pd.DataFrame:
        """Derive the circuit outline from a representative fast lap trace."""
        try:
            lap = session.laps.pick_fastest()
            tel = lap.get_telemetry()
            outline = tel.loc[:, ["X", "Y"]].copy().reset_index(drop=True)
            return outline
        except Exception:  # noqa: BLE001 - outline is best-effort
            return pd.DataFrame(columns=["X", "Y"])


def _safe_team_color(f1plotting, team: str, session) -> str:
    """Return a hex team colour, falling back gracefully across FastF1 versions."""
    if not team:
        return "#888888"
    # FastF1's plotting API changed across versions; try the known variants.
    for getter in (
        lambda: f1plotting.get_team_color(team, session=session),
        lambda: f1plotting.team_color(team),
        lambda: f1plotting.TEAM_COLORS[team.lower()],
    ):
        try:
            color = getter()
            if color:
                return str(color) if str(color).startswith("#") else f"#{color}"
        except Exception:  # noqa: BLE001 - try the next variant
            continue
    return "#888888"


def _friendly_error(prefix: str, exc: Exception) -> str:
    """Translate a raw loading exception into an actionable, user-facing string."""
    raw = str(exc)
    lowered = raw.lower()
    hint = ""
    if "401" in raw or "403" in raw or "unauthor" in lowered or "forbidden" in lowered:
        hint = (
            "\n\nThe F1 live-timing server rejected the request (HTTP 401/403). "
            "This is a known FastF1 issue (#978) that mostly affects recent "
            "2025/2026 sessions on a cold cache. Try a 2024 session (such as the "
            "default 2024 United States GP), or load the session once while you "
            "have a working connection so it is cached locally."
        )
    elif "connection" in lowered or "timed out" in lowered or "network" in lowered:
        hint = (
            "\n\nThis looks like a network problem. Check your internet "
            "connection; the first load of a session downloads 50-100MB."
        )
    return f"{prefix}{hint}\n\nTechnical detail: {raw}"


# Default source used by the app. Swap this for an OpenF1 source in future.
def default_source() -> SessionDataSource:
    """Return the default data source (FastF1 historical replay)."""
    return FastF1DataSource()
