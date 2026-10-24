"""Near-live F1 data source backed by the OpenF1 REST API.

OpenF1 (`<https://openf1.org>`_, base URL ``https://api.openf1.org/v1``) is a
free, no-authentication REST API that exposes **near-live** Formula 1 data with
roughly a 3 second delay. Car location updates arrive every few seconds while a
session is actually running ("green") on track.

This module provides :class:`OpenF1DataSource`, a concrete implementation of the
abstract :class:`~data.loader.SessionDataSource`. It returns the exact same
plain dataclasses (:class:`~data.loader.SessionData`,
:class:`~data.loader.DriverInfo`) that the FastF1 source returns, so the
visualisation layer (``viz/track_map.py``) and the Streamlit UI work against it
unchanged.

Design notes
------------
* HTTP is done with the standard library :mod:`urllib.request` so no new
  third-party dependency is introduced. A short timeout is used and every
  network/HTTP/JSON failure is converted into a
  :class:`~data.loader.SessionLoadError` carrying a friendly, user-facing
  message.
* The pure parsing / resampling helpers (``_build_*`` / ``resample_positions``)
  are deliberately written to accept plain Python lists of dicts (exactly the
  shape of OpenF1's JSON array responses) so they can be unit tested offline
  with small fake payloads, with no network and without importing FastF1.
* OpenF1 ``location`` coordinates live in their own self-consistent coordinate
  space (different from FastF1), which is fine: we draw the map from the data
  itself, so absolute units do not matter.

Field names below were implemented against the OpenF1 documentation. Because the
build/test environment has no network access, the exact field spellings could
not be verified live; the parsing is written defensively (``.get`` with
fallbacks) so a minor field rename degrades gracefully rather than crashing.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .loader import (
    DEFAULT_RESAMPLE_STEP_S,
    DriverInfo,
    SessionData,
    SessionDataSource,
    SessionLoadError,
)

OPENF1_BASE_URL = "https://api.openf1.org/v1"

# Short timeout so the UI stays responsive; live polling uses an even smaller
# window (see ``load_live_frame``).
DEFAULT_TIMEOUT_S = 15.0
LIVE_TIMEOUT_S = 6.0

# How far back (seconds) a live poll looks for "the most recent" sample per
# driver. Kept small so each poll is a tiny, fast query.
LIVE_WINDOW_S = 10

# OpenF1 session_name values that we expose, mapped to the shared session-type
# identifiers used across the app (see loader.SESSION_TYPES).
_SESSION_NAME_TO_TYPE = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Qualifying": "Q",
    "Sprint": "S",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SQ",
    "Race": "R",
}
_TYPE_TO_SESSION_NAME = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "Q": "Qualifying",
    "S": "Sprint",
    "SQ": "Sprint Qualifying",
    "R": "Race",
}


@dataclass
class LiveFrame:
    """A lightweight snapshot of the most recent position per driver.

    Returned by :meth:`OpenF1DataSource.load_live_frame` for the UI to poll. It
    is intentionally small (one point per driver) so each refresh is cheap.
    """

    session_key: int | None = None
    session_name: str | None = None
    # abbreviation -> (x, y)
    positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    # abbreviation -> DriverInfo (colour/name), so the UI can colour dots.
    drivers: dict[str, DriverInfo] = field(default_factory=dict)
    # ISO timestamp of the newest sample seen, for a "last update" caption.
    latest_ts: str | None = None
    # True when the session is on track and location data is flowing.
    is_live: bool = False
    # Optional standings: list of dicts {position, abbreviation, gap_to_leader}.
    standings: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _http_get_json(path: str, params: dict, timeout: float) -> list[dict]:
    """GET ``OPENF1_BASE_URL/path?params`` and parse the JSON array response.

    Returns a list of dicts (OpenF1 always returns a JSON array). Any network,
    HTTP or decoding failure is translated into a :class:`SessionLoadError`.
    """
    query = urllib.parse.urlencode(params, doseq=True)
    url = f"{OPENF1_BASE_URL}/{path.lstrip('/')}"
    if query:
        url = f"{url}?{query}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        raise SessionLoadError(
            _friendly_live_error(f"OpenF1 returned HTTP {exc.code} for /{path}.", exc)
        ) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network path
        raise SessionLoadError(
            _friendly_live_error("Could not reach the OpenF1 API.", exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001 - last resort, surfaced to the user
        raise SessionLoadError(
            _friendly_live_error("Unexpected error talking to OpenF1.", exc)
        ) from exc

    return _parse_json_array(payload, path)


def _parse_json_array(payload, path: str) -> list[dict]:
    """Decode an OpenF1 response body into a list of dicts.

    Split out from :func:`_http_get_json` so it can be unit tested offline.
    """
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        data = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise SessionLoadError(
            _friendly_live_error(f"OpenF1 sent an invalid response for /{path}.", exc)
        ) from exc
    if isinstance(data, dict):
        # Some error responses are a single object; wrap or surface it.
        if "detail" in data or "error" in data:
            raise SessionLoadError(
                f"OpenF1 reported an error for /{path}: "
                f"{data.get('detail') or data.get('error')}"
            )
        return [data]
    if not isinstance(data, list):
        return []
    return data


# ---------------------------------------------------------------------------
# Timestamp parsing (offline-testable)
# ---------------------------------------------------------------------------


def _parse_iso(ts: str) -> datetime | None:
    """Parse an OpenF1 ISO8601 timestamp into an aware ``datetime`` (UTC).

    OpenF1 timestamps look like ``2023-09-16T13:03:35.200000+00:00`` but may
    also end in ``Z`` or omit microseconds. Returns ``None`` if unparseable.
    """
    if not ts or not isinstance(ts, str):
        return None
    cleaned = ts.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        # Fall back to a few explicit formats.
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                dt = datetime.strptime(cleaned, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _seconds_from_first(timestamps: list[str]) -> list[float]:
    """Convert ISO timestamps to seconds-from-first-valid-sample.

    Unparseable entries map to ``nan``. The first *valid* timestamp is the zero
    reference.
    """
    parsed = [_parse_iso(t) for t in timestamps]
    first = next((p for p in parsed if p is not None), None)
    if first is None:
        return [float("nan")] * len(timestamps)
    return [
        (p - first).total_seconds() if p is not None else float("nan") for p in parsed
    ]


# ---------------------------------------------------------------------------
# Resampling (offline-testable, mirrors FastF1DataSource._build_positions)
# ---------------------------------------------------------------------------


def resample_positions(
    location_by_driver: dict[str, list[dict]],
    step_s: float = DEFAULT_RESAMPLE_STEP_S,
) -> tuple[np.ndarray, dict[str, pd.DataFrame]]:
    """Resample every driver's location trace onto a shared time axis.

    Parameters
    ----------
    location_by_driver:
        Mapping of driver abbreviation -> list of OpenF1 ``/location`` records
        (dicts with ``date``, ``x``, ``y``).
    step_s:
        Common time step (seconds).

    Returns
    -------
    ``(time_axis, positions)`` with the same shape/semantics as
    :meth:`FastF1DataSource._build_positions`: a shared numpy time axis and a
    mapping of abbreviation -> DataFrame indexed by that axis with ``X``/``Y``
    columns (NaN where the car has no sample).
    """
    per_driver_seconds: dict[str, pd.DataFrame] = {}
    global_max = 0.0

    for abbr, records in location_by_driver.items():
        if not records:
            continue
        dates = [r.get("date") for r in records]
        seconds = _seconds_from_first(dates)
        xs = [_to_float(r.get("x")) for r in records]
        ys = [_to_float(r.get("y")) for r in records]

        frame = pd.DataFrame({"t": seconds, "X": xs, "Y": ys})
        frame = frame.dropna(subset=["t"])
        if frame.empty:
            continue
        frame = frame.set_index("t")
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


def build_driver_info(driver_records: list[dict]) -> dict[str, DriverInfo]:
    """Build the abbreviation -> :class:`DriverInfo` map from /drivers records.

    OpenF1 ``team_colour`` is a hex string WITHOUT a leading ``#`` (e.g.
    ``"3671C6"``); we prefix ``#`` to match what the viz layer expects.
    """
    drivers: dict[str, DriverInfo] = {}
    for rec in driver_records or []:
        number = rec.get("driver_number")
        if number is None:
            continue
        abbr = rec.get("name_acronym") or rec.get("broadcast_name") or str(number)
        abbr = str(abbr)
        colour = rec.get("team_colour") or rec.get("team_color") or ""
        colour = str(colour).strip()
        if colour and not colour.startswith("#"):
            colour = f"#{colour}"
        if not colour:
            colour = "#888888"
        drivers[abbr] = DriverInfo(
            abbreviation=abbr,
            driver_number=str(number),
            full_name=str(rec.get("full_name") or rec.get("broadcast_name") or abbr),
            team_name=str(rec.get("team_name") or ""),
            team_color=colour,
        )
    return drivers


def build_track_outline(
    location_by_driver: dict[str, list[dict]],
) -> pd.DataFrame:
    """Derive a circuit outline from the driver with the densest location trace.

    A single driver's full lap trace traces the circuit, so we pick the driver
    with the most samples and use their ordered X/Y points.
    """
    best_abbr = None
    best_len = 0
    for abbr, records in location_by_driver.items():
        n = len(records or [])
        if n > best_len:
            best_len = n
            best_abbr = abbr
    if best_abbr is None:
        return pd.DataFrame(columns=["X", "Y"])
    records = location_by_driver[best_abbr]
    xs = [_to_float(r.get("x")) for r in records]
    ys = [_to_float(r.get("y")) for r in records]
    outline = pd.DataFrame({"X": xs, "Y": ys}).dropna().reset_index(drop=True)
    return outline


def index_drivers_by_number(
    drivers: dict[str, DriverInfo],
) -> dict[str, DriverInfo]:
    """Return a ``driver_number -> DriverInfo`` lookup."""
    return {info.driver_number: info for info in drivers.values()}


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------


class OpenF1DataSource(SessionDataSource):
    """Near-live data source backed by the OpenF1 REST API.

    Implements the same :class:`~data.loader.SessionDataSource` interface as
    :class:`~data.loader.FastF1DataSource`, so it drops into the existing UI and
    visualisation without structural changes.
    """

    def __init__(
        self,
        base_url: str = OPENF1_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- low level ------------------------------------------------------

    def _get(self, path: str, params: dict, timeout: float | None = None) -> list[dict]:
        return _http_get_json(path, params, timeout or self.timeout)

    # -- catalogue ------------------------------------------------------

    def list_years(self) -> list[int]:
        """Years with OpenF1 coverage (telemetry/location start in 2023)."""
        current = 2024
        try:
            current = max(current, datetime.now(timezone.utc).year)
        except Exception:  # pragma: no cover - defensive only
            pass
        return list(range(current, 2022, -1))

    def list_events(self, year: int) -> list[str]:
        """Distinct event (country) names for ``year`` from /sessions."""
        try:
            sessions = self._get("sessions", {"year": year})
        except SessionLoadError:
            raise
        names: list[str] = []
        seen: set[str] = set()
        for s in sessions:
            name = s.get("country_name") or s.get("circuit_short_name")
            if name and name not in seen:
                seen.add(name)
                names.append(str(name))
        return names

    def list_session_types(self) -> list[tuple[str, str]]:
        from .loader import SESSION_TYPES

        return list(SESSION_TYPES)

    # -- session resolution --------------------------------------------

    def _find_session(self, year: int, event: str, session_type: str) -> dict:
        """Resolve a single OpenF1 session dict for (year, event, type)."""
        session_name = _TYPE_TO_SESSION_NAME.get(session_type)
        params: dict = {"year": year, "country_name": event}
        if session_name:
            params["session_name"] = session_name
        sessions = self._get("sessions", params)
        if not sessions:
            # Retry without the session_name filter in case of a naming
            # mismatch, then match client-side.
            sessions = self._get("sessions", {"year": year, "country_name": event})
            if session_name:
                sessions = [
                    s for s in sessions if s.get("session_name") == session_name
                ] or sessions
        if not sessions:
            raise SessionLoadError(
                f"No OpenF1 session found for {year} {event} "
                f"({session_type}). Try a different event or session type, "
                f"or use Replay mode."
            )
        # Prefer the latest matching session by date_start.
        sessions.sort(key=lambda s: str(s.get("date_start") or ""))
        return sessions[-1]

    def resolve_latest_session(self) -> dict | None:
        """Return the most recent/ongoing OpenF1 session, or ``None``.

        Uses ``session_key=latest`` against /sessions. Returns ``None`` if the
        API yields nothing (rather than raising) so the UI can show a friendly
        "no live session" message.
        """
        try:
            sessions = self._get(
                "sessions", {"session_key": "latest"}, timeout=LIVE_TIMEOUT_S
            )
        except SessionLoadError:
            raise
        if not sessions:
            return None
        return sessions[0]

    # -- loading --------------------------------------------------------

    def load(
        self,
        year: int,
        event: str,
        session_type: str,
        resample_step_s: float = DEFAULT_RESAMPLE_STEP_S,
    ) -> SessionData:
        """Load a completed-or-ongoing OpenF1 session into a ``SessionData``.

        Fetches the drivers and full location history for the session, builds a
        shared time axis, resamples each driver's X/Y onto it, and derives a
        track outline, exactly like :class:`FastF1DataSource`.
        """
        session = self._find_session(year, event, session_type)
        session_key = session.get("session_key")
        if session_key is None:
            raise SessionLoadError(
                f"OpenF1 session for {year} {event} ({session_type}) has no "
                f"session_key; cannot load it."
            )

        driver_records = self._get("drivers", {"session_key": session_key})
        drivers = build_driver_info(driver_records)
        if not drivers:
            raise SessionLoadError(
                f"OpenF1 returned no drivers for {year} {event} ({session_type})."
            )

        location_by_driver: dict[str, list[dict]] = {}
        for abbr, info in drivers.items():
            records = self._get(
                "location",
                {"session_key": session_key, "driver_number": info.driver_number},
            )
            # OpenF1 may include (0,0) placeholder samples before green; drop
            # exact origin points which are not real track positions.
            records = [
                r
                for r in records
                if not (_to_float(r.get("x")) == 0.0 and _to_float(r.get("y")) == 0.0)
            ]
            if records:
                location_by_driver[abbr] = records

        if not location_by_driver:
            raise SessionLoadError(
                f"OpenF1 has no location data yet for {year} {event} "
                f"({session_type}). Location data only exists once a session has "
                f"run on track. Try a completed session, or use Replay mode."
            )

        time_axis, positions = resample_positions(location_by_driver, resample_step_s)
        track_outline = build_track_outline(location_by_driver)

        meta = {
            "year": year,
            "event": event,
            "session_type": session_type,
            "session_name": session.get("session_name", session_type),
            "event_name": session.get("country_name", event),
            "session_key": session_key,
            "source": "openf1",
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
            raw_session=None,
        )

    # -- live polling ---------------------------------------------------

    def load_live_frame(
        self,
        session_key="latest",
        window_s: int = LIVE_WINDOW_S,
        drivers: dict[str, DriverInfo] | None = None,
    ) -> LiveFrame:
        """Fetch the most recent position per driver for near-live updates.

        Parameters
        ----------
        session_key:
            A concrete session_key, or the literal string ``"latest"`` to target
            the current/most-recent session.
        window_s:
            How many seconds back to look for samples. Kept small so the query
            is fast and the UI stays responsive.
        drivers:
            Optional pre-fetched driver metadata to avoid re-querying /drivers
            on every poll. If omitted it is fetched once here.

        Returns
        -------
        LiveFrame
            A small snapshot. ``is_live`` is ``False`` (with empty positions) if
            no location samples arrived in the window, which the UI uses to show
            a clear "no live session" message instead of a blank screen.
        """
        # Resolve the concrete session.
        if session_key == "latest":
            session = self.resolve_latest_session()
            if session is None:
                return LiveFrame(is_live=False)
            resolved_key = session.get("session_key")
            session_name = session.get("session_name")
        else:
            resolved_key = session_key
            session_name = None

        if resolved_key is None:
            return LiveFrame(is_live=False, session_name=session_name)

        # Driver metadata (colours/abbreviations), fetched once if not provided.
        if drivers is None:
            try:
                driver_records = self._get(
                    "drivers", {"session_key": resolved_key}, timeout=LIVE_TIMEOUT_S
                )
            except SessionLoadError:
                driver_records = []
            drivers = build_driver_info(driver_records)
        by_number = index_drivers_by_number(drivers)

        # Small, recent location window across all drivers in the session.
        since = _iso_utc_now_minus(window_s)
        records = self._get(
            "location",
            {"session_key": resolved_key, "date>": since},
            timeout=LIVE_TIMEOUT_S,
        )

        latest_by_driver = latest_position_per_driver(records, by_number)
        positions: dict[str, tuple[float, float]] = {}
        latest_ts: str | None = None
        for number, (x, y, ts) in latest_by_driver.items():
            info = by_number.get(str(number))
            abbr = info.abbreviation if info else str(number)
            positions[abbr] = (x, y)
            if ts and (latest_ts is None or ts > latest_ts):
                latest_ts = ts

        standings = self._maybe_standings(resolved_key, by_number)

        return LiveFrame(
            session_key=resolved_key,
            session_name=session_name,
            positions=positions,
            drivers=drivers,
            latest_ts=latest_ts,
            is_live=bool(positions),
            standings=standings,
        )

    def _maybe_standings(
        self, session_key, by_number: dict[str, DriverInfo]
    ) -> list[dict]:
        """Best-effort live standings from /position + /intervals.

        Never raises: on any failure it returns an empty list so live polling of
        positions is never blocked by the standings extra.
        """
        try:
            pos_records = self._get(
                "position", {"session_key": session_key}, timeout=LIVE_TIMEOUT_S
            )
        except SessionLoadError:
            return []
        latest_pos = latest_position_standings(pos_records)
        if not latest_pos:
            return []

        gaps: dict[str, str] = {}
        try:
            interval_records = self._get(
                "intervals", {"session_key": session_key}, timeout=LIVE_TIMEOUT_S
            )
            gaps = latest_gap_to_leader(interval_records)
        except SessionLoadError:
            gaps = {}

        standings: list[dict] = []
        for number, position in sorted(latest_pos.items(), key=lambda kv: kv[1]):
            info = by_number.get(str(number))
            standings.append(
                {
                    "position": position,
                    "abbreviation": info.abbreviation if info else str(number),
                    "gap_to_leader": gaps.get(str(number)),
                }
            )
        return standings


# ---------------------------------------------------------------------------
# Live aggregation helpers (offline-testable)
# ---------------------------------------------------------------------------


def latest_position_per_driver(
    location_records: list[dict],
    by_number: dict[str, DriverInfo] | None = None,
) -> dict[str, tuple[float, float, str | None]]:
    """Reduce raw /location records to the newest (x, y, date) per driver.

    Keyed by ``driver_number`` (as a string). (0,0) origin placeholders and rows
    without a usable date are ignored.
    """
    newest: dict[str, tuple[float, float, str | None]] = {}
    newest_dt: dict[str, datetime] = {}
    for rec in location_records or []:
        number = rec.get("driver_number")
        if number is None:
            continue
        number = str(number)
        x = _to_float(rec.get("x"))
        y = _to_float(rec.get("y"))
        if x is None or y is None:
            continue
        if x == 0.0 and y == 0.0:
            continue
        ts = rec.get("date")
        dt = _parse_iso(ts) if ts else None
        if dt is None:
            # Keep as a candidate only if we have nothing for this driver.
            if number not in newest:
                newest[number] = (x, y, ts)
            continue
        if number not in newest_dt or dt > newest_dt[number]:
            newest_dt[number] = dt
            newest[number] = (x, y, ts)
    return newest


def latest_position_standings(position_records: list[dict]) -> dict[str, int]:
    """Reduce /position records to the newest ``position`` per driver number."""
    newest: dict[str, int] = {}
    newest_dt: dict[str, datetime] = {}
    for rec in position_records or []:
        number = rec.get("driver_number")
        position = rec.get("position")
        if number is None or position is None:
            continue
        number = str(number)
        dt = _parse_iso(rec.get("date")) if rec.get("date") else None
        try:
            position = int(position)
        except (TypeError, ValueError):
            continue
        if dt is None:
            newest.setdefault(number, position)
            continue
        if number not in newest_dt or dt > newest_dt[number]:
            newest_dt[number] = dt
            newest[number] = position
    return newest


def latest_gap_to_leader(interval_records: list[dict]) -> dict[str, str]:
    """Reduce /intervals records to the newest ``gap_to_leader`` per driver."""
    newest: dict[str, str] = {}
    newest_dt: dict[str, datetime] = {}
    for rec in interval_records or []:
        number = rec.get("driver_number")
        if number is None:
            continue
        number = str(number)
        gap = rec.get("gap_to_leader")
        dt = _parse_iso(rec.get("date")) if rec.get("date") else None
        gap_str = "" if gap is None else str(gap)
        if dt is None:
            newest.setdefault(number, gap_str)
            continue
        if number not in newest_dt or dt > newest_dt[number]:
            newest_dt[number] = dt
            newest[number] = gap_str
    return newest


# ---------------------------------------------------------------------------
# misc helpers
# ---------------------------------------------------------------------------


def _to_float(value):
    """Best-effort float conversion; returns ``None`` on failure."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _iso_utc_now_minus(seconds: int) -> str:
    """Return an ISO8601 UTC timestamp ``seconds`` in the past (for date> filter)."""
    now = datetime.now(timezone.utc)
    past_epoch = now.timestamp() - max(0, seconds)
    return datetime.fromtimestamp(past_epoch, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )


def _friendly_live_error(prefix: str, exc: Exception) -> str:
    """Translate an OpenF1/network exception into a user-facing message."""
    raw = str(exc)
    lowered = raw.lower()
    hint = ""
    if "timed out" in lowered or "timeout" in lowered:
        hint = (
            "\n\nThe request to OpenF1 timed out. Check your internet connection "
            "and try again; live polling uses a short timeout to stay responsive."
        )
    elif "name or service not known" in lowered or "getaddrinfo" in lowered:
        hint = (
            "\n\nCould not resolve api.openf1.org. Are you online? OpenF1 is a "
            "public API and needs internet access."
        )
    elif "404" in raw:
        hint = (
            "\n\nThat OpenF1 endpoint or session was not found. The session may "
            "not exist yet, or a field/name may be slightly off."
        )
    return f"{prefix}{hint}\n\nTechnical detail: {raw}"


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def live_source() -> OpenF1DataSource:
    """Return the near-live OpenF1 data source."""
    return OpenF1DataSource()
