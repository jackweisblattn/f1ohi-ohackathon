"""Historical F1 telemetry replay powered by FastF1. Run with: streamlit run app.py"""
# Inspired by Tom Shaw's F1 race replay visualization: https://www.youtube.com/watch?v=TiQEElXyY2w
from pathlib import Path
import base64
import json
import math
import re
from collections import defaultdict
from bisect import bisect_left
from io import StringIO

import fastf1
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Trackside Replay", page_icon="🏁", layout="wide")
CACHE_DIR = Path(__file__).resolve().parent / "work" / "fastf1_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.configure(cache_dir=str(CACHE_DIR))


def safe_number(value):
    return None if pd.isna(value) else float(value)


@st.cache_data(ttl=86400, show_spinner="Loading historical F1 data… The first load can take a few minutes.")
def get_events(year):
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    # Keep only actual, numbered Grand Prix race sessions. FastF1's schedule is
    # season-specific; requiring a race session and a real past start time also
    # excludes testing rows, incomplete calendar entries, and future rounds.
    if schedule.empty or "Session5DateUtc" not in schedule:
        return []
    round_numbers = pd.to_numeric(schedule.get("RoundNumber"), errors="coerce")
    race_names = schedule.get("Session5", pd.Series("", index=schedule.index)).fillna("").astype(str).str.strip().str.casefold()
    event_names = schedule.get("EventName", pd.Series("", index=schedule.index)).fillna("").astype(str).str.strip()
    locations = schedule.get("Location", pd.Series("", index=schedule.index)).fillna("").astype(str).str.strip()
    race_dates = pd.to_datetime(schedule["Session5DateUtc"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    valid = (
        round_numbers.gt(0)
        & race_names.eq("race")
        & event_names.ne("")
        & locations.ne("")
        & race_dates.notna()
        & race_dates.le(now)
    )
    events = schedule.loc[valid, ["RoundNumber", "EventName", "Location"]].copy()
    events["RoundNumber"] = round_numbers.loc[valid].astype(int)
    # Avoid duplicate calendar entries while preserving the official round order.
    events = events.drop_duplicates(subset=["RoundNumber", "EventName"])
    return events[["RoundNumber", "EventName", "Location"]].to_dict("records")


@st.cache_data(ttl=86400, show_spinner=False)
def get_dotd_records():
    """Download the attributed Driver of the Day results once per day."""
    base = "https://raw.githubusercontent.com/TracingInsights/RaceData/main/f1db-csv"
    awards_response = requests.get(
        f"{base}/f1db-races-driver-of-the-day-results.csv", timeout=10
    )
    awards_response.raise_for_status()
    drivers_response = requests.get(f"{base}/f1db-drivers.csv", timeout=10)
    drivers_response.raise_for_status()
    awards = pd.read_csv(
        StringIO(awards_response.text),
        usecols=["year", "round", "driverId"],
    )
    drivers = pd.read_csv(
        StringIO(drivers_response.text),
        usecols=["id", "fullName", "abbreviation"],
    )
    merged = awards.merge(drivers, left_on="driverId", right_on="id", how="inner")
    return merged[["year", "round", "fullName", "abbreviation"]].to_dict("records")


def get_official_dotd(year, round_number):
    """Look up the fan-voted award winner for a selected season round."""
    records = get_dotd_records()
    matches = [
        row for row in records
        if int(row["year"]) == int(year) and int(row["round"]) == int(round_number)
    ]
    if not matches:
        return None
    return {
        "name": str(matches[0]["fullName"]),
        "abbr": str(matches[0]["abbreviation"]),
    }


@st.cache_data(ttl=86400, show_spinner="Loading season championship points…")
def get_championship_data(year):
    """Build round-by-round driver and constructor points from classified results."""
    completed_events = get_events(year)
    if not completed_events:
        return []
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    event_by_round = {int(event["RoundNumber"]): event for event in completed_events}
    driver_points, constructor_points = defaultdict(float), defaultdict(float)
    driver_meta, constructor_meta, snapshots = {}, {}, []

    for round_number in sorted(event_by_round):
        event = event_by_round[round_number]
        event_rows = schedule[pd.to_numeric(schedule["RoundNumber"], errors="coerce").eq(round_number)]
        sprint_codes = []
        if not event_rows.empty:
            row = event_rows.iloc[0]
            for index in range(1, 6):
                if str(row.get(f"Session{index}", "")).strip().casefold() == "sprint":
                    sprint_codes.append("S")

        for code in sprint_codes + ["R"]:
            try:
                session = fastf1.get_session(int(year), round_number, code)
                session.load(telemetry=False, weather=False, messages=False)
                results = session.results.copy()
            except Exception:
                # Some old or partially published sessions have no result feed.
                continue
            if results.empty or "Points" not in results:
                continue
            for _, result in results.iterrows():
                number = str(result.get("DriverNumber", "")).strip()
                name = str(result.get("FullName", result.get("Abbreviation", number))).strip()
                abbreviation = str(result.get("Abbreviation", number)).strip()
                team = str(result.get("TeamName", "Unknown team")).strip()
                colour = str(result.get("TeamColor", "8290a2")).strip().lstrip("#")
                points = safe_number(result.get("Points")) or 0.0
                if not number:
                    continue
                driver_points[number] += points
                constructor_points[team] += points
                driver_meta[number] = {"name": name, "abbr": abbreviation, "team": team, "colour": colour}
                constructor_meta[team] = {"colour": colour}

        snapshots.append({
            "round": round_number,
            "event": event["EventName"],
            "drivers": [
                {"number": number, **driver_meta[number], "points": points}
                for number, points in sorted(driver_points.items(), key=lambda item: (-item[1], driver_meta[item[0]]["name"]))
            ],
            "constructors": [
                {"team": team, "colour": constructor_meta[team]["colour"], "points": points}
                for team, points in sorted(constructor_points.items(), key=lambda item: (-item[1], item[0]))
            ],
        })
    return snapshots


@st.cache_data(ttl=86400, show_spinner="Downloading the recorded session. FastF1 will cache it for next time.")
def load_replay(year, round_number, session_code, qualifying_round=None):
    session = fastf1.get_session(int(year), int(round_number), session_code)
    session.load(telemetry=True, weather=False, messages=True)

    def to_session_seconds(values):
        """Handle FastF1 time columns represented as durations or absolute datetimes."""
        series = values if isinstance(values, pd.Series) else pd.Series(values)
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            timestamps = pd.to_datetime(series, utc=True)
            origin = session.t0_date
            if origin is not None and not pd.isna(origin):
                return (timestamps - pd.to_datetime(origin, utc=True)).dt.total_seconds()
            return (timestamps - timestamps.min()).dt.total_seconds()
        if pd.api.types.is_timedelta64_dtype(series.dtype):
            return series.dt.total_seconds()
        return pd.to_timedelta(series, errors="coerce").dt.total_seconds()

    position_rows = []
    for driver_number, data in session.pos_data.items():
        if data is None or data.empty:
            continue
        data = data.copy()
        data["driver_number"] = int(driver_number)
        if "SessionTime" in data:
            data["seconds"] = to_session_seconds(data["SessionTime"])
        elif "Date" in data:
            data["Date"] = pd.to_datetime(data["Date"], utc=True)
            data["seconds"] = (data["Date"] - data["Date"].min()).dt.total_seconds()
        position_rows.append(data)

    locations = pd.concat(position_rows, ignore_index=True) if position_rows else pd.DataFrame()
    if not locations.empty:
        locations = locations.dropna(subset=["X", "Y", "seconds"]).sort_values("seconds")

    telemetry_rows = []
    for driver_number, data in (getattr(session, "car_data", {}) or {}).items():
        if data is None or data.empty:
            continue
        data = data.copy()
        data["driver_number"] = int(driver_number)
        if "SessionTime" in data:
            data["seconds"] = to_session_seconds(data["SessionTime"])
        elif "Date" in data:
            dates = pd.to_datetime(data["Date"], utc=True)
            data["seconds"] = (dates - dates.min()).dt.total_seconds()
        else:
            continue
        normalized_columns = {
            re.sub(r"[^a-z]", "", str(column).lower()): column
            for column in data.columns
        }
        aero_column = next(
            (normalized_columns[name] for name in ("activeaero", "activeaeromode", "aeromode") if name in normalized_columns),
            None,
        )
        data["active_aero"] = data[aero_column] if aero_column else pd.NA
        telemetry_rows.append(data)
    car_telemetry = pd.concat(telemetry_rows, ignore_index=True) if telemetry_rows else pd.DataFrame()
    if not car_telemetry.empty:
        car_telemetry = car_telemetry.dropna(subset=["seconds"]).sort_values("seconds")
        # 2 Hz is enough for an overview chart and keeps the browser payload small.
        car_telemetry["bucket"] = (car_telemetry["seconds"] * 2).astype(int)
        car_telemetry = car_telemetry.drop_duplicates(["driver_number", "bucket"], keep="last")

    results = session.results.copy()
    drivers = pd.DataFrame({
        "driver_number": results["DriverNumber"].astype(int),
        "name_acronym": results["Abbreviation"].fillna(""),
        "full_name": results.get("FullName", results["Abbreviation"]).fillna(""),
        "team_name": results["TeamName"].fillna(""),
        "team_colour": results["TeamColor"].fillna("FFFFFF"),
        "grid_position": results["GridPosition"],
        "headshot_url": results["HeadshotUrl"].fillna("") if "HeadshotUrl" in results else "",
        "result_status": results["Status"].fillna("") if "Status" in results else "",
    })

    laps = session.laps.copy()
    if not laps.empty:
        laps["driver_number"] = laps["DriverNumber"].astype(int)
        if "Time" in laps:
            laps["seconds"] = to_session_seconds(laps["Time"])
        else:
            laps["seconds"] = pd.NA
        if "LapStartTime" in laps:
            laps["lap_start_seconds"] = to_session_seconds(laps["LapStartTime"])
        else:
            laps["lap_start_seconds"] = pd.NA
        if "LapTime" in laps:
            laps["lap_seconds"] = to_session_seconds(laps["LapTime"])
        for column in ("PitInTime", "PitOutTime"):
            if column in laps:
                laps[f"{column}_seconds"] = to_session_seconds(laps[column])
        for sector in (1, 2, 3):
            column = f"Sector{sector}Time"
            laps[f"sector{sector}_seconds"] = to_session_seconds(laps[column]) if column in laps else pd.NA

    track_status = session.track_status.copy()
    if not track_status.empty and "Time" in track_status:
        track_status["seconds"] = to_session_seconds(track_status["Time"])
    race_control = session.race_control_messages.copy()
    if not race_control.empty and "Time" in race_control:
        race_control["seconds"] = to_session_seconds(race_control["Time"])

    segment_window = None
    if session_code == "Q" and qualifying_round:
        round_index = int(qualifying_round[-1]) - 1
        split_laps = laps.split_qualifying_sessions()
        if round_index >= len(split_laps) or split_laps[round_index] is None:
            raise ValueError(f"{qualifying_round} data is unavailable for this event.")

        split_points = list(getattr(session, "_session_split_times", []) or [])
        split_seconds = pd.to_numeric(
            to_session_seconds(pd.Series(split_points)), errors="coerce"
        ).dropna().tolist()
        session_end = (
            safe_number(to_session_seconds(session.session_status["Time"].iloc[-1:]).iloc[0])
            if session.session_status is not None and not session.session_status.empty
            else safe_number(locations["seconds"].max())
        )
        split_seconds = [value for value in split_seconds if value < 86400]
        if len(split_seconds) < 3:
            starts = []
            suspended = False
            if session.session_status is not None:
                for _, status in session.session_status.iterrows():
                    if status["Status"] == "Started":
                        if not suspended:
                            starts.append(safe_number(to_session_seconds(pd.Series([status["Time"]])).iloc[0]))
                        suspended = False
                    elif status["Status"] == "Aborted":
                        suspended = True
                    elif status["Status"] == "Finished":
                        suspended = False
            split_seconds = [value for value in starts if value is not None]
        if round_index >= len(split_seconds):
            raise ValueError(f"Could not determine the {qualifying_round} time window.")

        segment_start = split_seconds[round_index]
        segment_end = split_seconds[round_index + 1] if round_index + 1 < len(split_seconds) else session_end
        if segment_end is None or segment_end <= segment_start:
            segment_end = safe_number(split_laps[round_index]["Time"].max())
        segment_window = (segment_start, segment_end)

        round_laps = split_laps[round_index]
        participants = set(round_laps["DriverNumber"].dropna().astype(int))
        if not locations.empty:
            round_positions = locations["seconds"].between(segment_start, segment_end)
            locations = locations.loc[
                round_positions & locations["driver_number"].isin(participants)
            ].copy()
        drivers = drivers[drivers["driver_number"].isin(participants)].copy()
        laps = round_laps.copy()
        if not car_telemetry.empty:
            car_telemetry = car_telemetry[
                car_telemetry["seconds"].between(segment_start, segment_end)
                & car_telemetry["driver_number"].isin(participants)
            ].copy()
        if not track_status.empty:
            track_status = track_status[track_status["seconds"].between(segment_start, segment_end)].copy()
        if not race_control.empty:
            race_control = race_control[race_control["seconds"].between(segment_start, segment_end)].copy()

    total_laps = session.total_laps
    total_laps = int(total_laps) if total_laps is not None and not pd.isna(total_laps) else None
    return locations, drivers, laps, track_status, race_control, car_telemetry, total_laps, segment_window


st.title("🏁 Trackside Replay")
st.caption("Replay recorded Grand Prix sessions from 2025 onward · Historical data via FastF1")

with st.sidebar:
    st.header("Choose a replay")
    current_season = pd.Timestamp.now().year
    seasons = list(range(current_season, 2017, -1))
    year = st.selectbox("Season", seasons, index=0)

try:
    events = get_events(year)
except Exception as exc:
    st.error(f"Couldn't get the F1 event schedule: {exc}")
    st.stop()
if not events:
    st.warning("No completed Grand Prix events with replay data were found for this season.")
    st.stop()

with st.sidebar:
    event = st.selectbox(
        "Grand Prix",
        events,
        format_func=lambda item: f"Round {int(item['RoundNumber'])} · {item['EventName']}",
    )
    session_name = st.selectbox("Session", ["Race", "Qualifying · Q1", "Qualifying · Q2", "Qualifying · Q3", "Practice 1", "Practice 2", "Practice 3"])
    qualifying_round = session_name.rsplit(" ", 1)[-1] if session_name.startswith("Qualifying") else None
    session_code = "Q" if qualifying_round else {"Race": "R", "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3"}[session_name]

try:
    locations, drivers, laps, track_status, race_control, car_telemetry, total_laps, segment_window = load_replay(
        year, event["RoundNumber"], session_code, qualifying_round
    )
except Exception as exc:
    st.error(f"Couldn't load this historical session: {exc}")
    st.stop()
if locations.empty or drivers.empty:
    st.warning("This session has no recorded position data.")
    st.stop()

locations["driver_number"] = locations["driver_number"].astype(int)
drivers["driver_number"] = drivers["driver_number"].astype(int)

max_seconds = max(1, int(locations["seconds"].max()))
driver_starts = locations.groupby("driver_number")["seconds"].min()


def detect_race_start(locations, race_control, car_telemetry):
    """Find lights out so a race replay opens at the start instead of the grid build-up."""
    if not race_control.empty and {"seconds", "Message"}.issubset(race_control.columns):
        messages = race_control.dropna(subset=["seconds"]).copy()
        lights_out = messages[
            messages["Message"].fillna("").astype(str).str.contains(
                r"lights?\s*out|green\s*light|race\s*start(?:ed)?",
                case=False,
                regex=True,
            )
        ]
        if not lights_out.empty:
            return safe_number(lights_out.iloc[0]["seconds"])

    # Speed telemetry can locate the launch more precisely than position samples.
    if not car_telemetry.empty and {"driver_number", "seconds", "Speed"}.issubset(car_telemetry.columns):
        moving = car_telemetry[pd.to_numeric(car_telemetry["Speed"], errors="coerce") >= 15]
        first_moves = (
            moving.groupby("driver_number")["seconds"].min().dropna().astype(float).tolist()
        )
        if len(first_moves) >= 5:
            first_moves.sort()
            required = max(5, int(len(first_moves) * 0.5))
            for index, start in enumerate(first_moves):
                if sum(value <= start + 8 for value in first_moves[index:]) >= required:
                    return start

    if locations.empty or not {"driver_number", "seconds", "X", "Y"}.issubset(locations.columns):
        return None
    first_moves = []
    for _, samples in locations.groupby("driver_number", sort=False):
        samples = samples.sort_values("seconds")
        if len(samples) < 2:
            continue
        first = samples.iloc[0]
        distance = ((samples["X"] - first["X"]) ** 2 + (samples["Y"] - first["Y"]) ** 2) ** 0.5
        moved = samples.loc[distance >= 15, "seconds"]
        if not moved.empty:
            first_moves.append(float(moved.iloc[0]))
    if len(first_moves) < 5:
        return None
    first_moves.sort()
    # The launch is coordinated; avoid mistaking a single car leaving the pits for it.
    required = max(5, int(len(first_moves) * 0.5))
    for index, start in enumerate(first_moves):
        clustered = [value for value in first_moves[index:] if value <= start + 8]
        if len(clustered) >= required:
            return start
    return None


if segment_window:
    replay_start = float(segment_window[0])
    max_seconds = max(max_seconds, int(segment_window[1]))
else:
    replay_start = float(driver_starts.quantile(0.8)) if not driver_starts.empty else 0.0
    if session_code == "R":
        detected_start = detect_race_start(locations, race_control, car_telemetry)
        if detected_start is not None:
            replay_start = detected_start
replay_duration = max(1.0, max_seconds - replay_start)

# Use one completed lap as the map outline.
track = pd.DataFrame()
if not laps.empty and {"LapStartTime", "Time", "LapTime", "DriverNumber"}.issubset(laps.columns):
    clean_laps = laps.dropna(subset=["LapStartTime", "Time", "LapTime"]).sort_values("LapTime")
    for _, reference in clean_laps.iterrows():
        ref_driver = int(reference["DriverNumber"])
        candidate = locations[
            (locations["driver_number"] == ref_driver)
            & locations["seconds"].between(float(reference["lap_start_seconds"]), float(reference["seconds"]))
        ].sort_values("seconds")
        candidate = candidate[
            candidate["X"].notna() & candidate["Y"].notna()
            & ((candidate["X"] != 0) | (candidate["Y"] != 0))
        ]
        # FastF1 position coordinates are in 1/10 m units. Reject partial
        # laps and laps with implausible jumps rather than drawing them as a map.
        if len(candidate) >= 100:
            coords = candidate[["X", "Y"]].to_numpy(dtype=float)
            distance = float(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])).sum())
            sample_gaps = candidate["seconds"].diff().dropna().to_numpy(dtype=float)
            sample_steps = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
            plausible_steps = np.all(
                (sample_gaps <= 5) & (sample_steps <= np.maximum(2500, sample_gaps * 1500 + 500))
            )
            closes = math.hypot(*(coords[-1] - coords[0])) <= 2000
            if 25_000 <= distance <= 100_000 and plausible_steps and closes:
                track = candidate
                break


def track_from_position_loop(locations):
    """Recover a closed, full circuit lap when lap and position clocks differ."""
    if locations.empty or not {"driver_number", "X", "Y"}.issubset(locations.columns):
        return pd.DataFrame()
    best_candidate = None
    for _, samples in locations.groupby("driver_number", sort=False):
        samples = samples.sort_values("seconds").dropna(subset=["X", "Y"]).copy()
        samples = samples[(samples["X"] != 0) | (samples["Y"] != 0)].reset_index(drop=True)
        if len(samples) < 100 or "seconds" not in samples:
            continue
        coords = samples[["X", "Y"]].to_numpy(dtype=float)
        seconds = samples["seconds"].to_numpy(dtype=float)
        steps = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
        elapsed = np.diff(seconds)
        # Break at missing intervals and impossible jumps so unrelated sections
        # of telemetry can never be joined into a fake circuit outline.
        breaks = np.flatnonzero((elapsed <= 0) | (elapsed > 5) | (steps > np.maximum(2500, elapsed * 1500 + 500))) + 1
        boundaries = np.concatenate(([0], breaks, [len(samples)]))
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            part = samples.iloc[left:right].reset_index(drop=True)
            if len(part) < 80:
                continue
            points = part[["X", "Y"]].to_numpy(dtype=float)
            times = part["seconds"].to_numpy(dtype=float)
            step = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
            cumulative = np.concatenate(([0.0], np.cumsum(step)))
            # FastF1 X/Y are decimetres: a full lap is roughly 25k–100k units.
            for start in range(0, len(part) - 50, 8):
                low = bisect_left(cumulative, cumulative[start] + 25_000, lo=start + 1)
                high = bisect_left(cumulative, cumulative[start] + 100_000, lo=low)
                for end in range(max(start + 40, low), min(high, len(part)), 3):
                    duration = times[end] - times[start]
                    if duration < 45 or duration > 180:
                        continue
                    distance = cumulative[end] - cumulative[start]
                    closure = math.hypot(
                        points[end, 0] - points[start, 0],
                        points[end, 1] - points[start, 1],
                    )
                    if closure <= 500:
                        # Prefer a short, realistic lap that closes cleanly.
                        score = closure + abs(duration - 90) * 4 + distance * 0.0001
                        rank = (score, distance)
                        if best_candidate is None or rank < best_candidate[:2]:
                            best_candidate = (score, distance, part, start, end)
    if best_candidate is None:
        return pd.DataFrame()
    return best_candidate[2].iloc[best_candidate[3]:best_candidate[4] + 1].copy()


if len(track) < 100:
    track = track_from_position_loop(locations)
if not track.empty:
    track = track.iloc[::max(1, len(track) // 900)]


def dnf_info(driver_number, result_status, driver_laps):
    """Use the official result to identify DNFs, never pit visits or temporary missing samples."""
    status = "" if pd.isna(result_status) else str(result_status).strip()
    if session_code != "R" or not status:
        return None
    # FastF1 labels classified finishers who are a lap down as "Lapped".
    if status.casefold() in {"finished", "lapped"} or re.fullmatch(r"\+\s*\d+\s+Lap(?:s)?", status, re.I):
        return None

    # Position streams may keep repeating a parked car's coordinates long
    # after retirement, so use an incident timestamp or last completed lap.
    stopped_at = None
    reason = status
    lap_stopped = None

    matching = race_control
    if not matching.empty and "RacingNumber" in matching:
        matching = matching[matching["RacingNumber"].astype(str) == str(driver_number)]
    if not matching.empty and "Message" in matching:
        text = matching["Message"].fillna("").astype(str)
        incident = matching[
            text.str.contains(r"retir|stopp|crash|collision|accident|engine|gearbox|damage|spin|contact|hydraulic|brake|power unit", case=False, regex=True)
            & ~text.str.contains("pit", case=False, regex=False)
        ]
        if not incident.empty:
            message = incident.iloc[-1]
            reason = str(message["Message"]).strip() or status
            lap_stopped = safe_number(message.get("Lap"))
            message_time = safe_number(message.get("seconds"))
            if message_time is not None:
                stopped_at = message_time

    completed = driver_laps.dropna(subset=["seconds", "lap_seconds"]).sort_values("seconds") if not driver_laps.empty and "lap_seconds" in driver_laps else pd.DataFrame()
    if stopped_at is None and not completed.empty:
        stopped_at = safe_number(completed.iloc[-1]["seconds"])
    if not completed.empty and stopped_at is not None:
        before_stop = completed[completed["seconds"] <= stopped_at]
        if not before_stop.empty:
            # The race-control Lap is often the lap in which the incident
            # happened; this is the last lap the driver actually completed.
            lap_stopped = safe_number(before_stop.iloc[-1].get("LapNumber"))
    # FastF1 may include stationary grid/pit position samples for DNS cars.
    # Their official result means they are out from the start, regardless of
    # those samples' timestamps.
    if re.search(r"did not start|\bDNS\b", status, re.I):
        stopped_at = 0.0
        lap_stopped = None
    if stopped_at is None:
        # A retired car may have no completed lap at all. In that case its
        # only lap row marks its first/aborted attempt; use that timestamp
        # rather than the end of the whole session (which would keep it in
        # the live tower with a bogus interpolated gap for most of the race).
        if not driver_laps.empty and "seconds" in driver_laps:
            stopped_at = safe_number(driver_laps["seconds"].min())
        if stopped_at is None:
            driver_positions = locations[locations["driver_number"] == int(driver_number)]
            if not driver_positions.empty:
                stopped_at = safe_number(driver_positions["seconds"].min())
        if stopped_at is None:
            stopped_at = safe_number(locations["seconds"].max())
        if lap_stopped is None and stopped_at is not None and completed.empty:
            lap_stopped = 0.0
    return {"time": stopped_at, "lap": lap_stopped, "reason": reason, "status": status}


def crash_info(driver_number, driver_laps):
    """Hide a practice/qualifying marker only after an explicit crash report."""
    if race_control.empty or "Message" not in race_control or "RacingNumber" not in race_control:
        return None
    numbers = race_control["RacingNumber"].astype("string").str.replace(r"\.0$", "", regex=True)
    matching = race_control[numbers == str(driver_number)].copy()
    if matching.empty:
        return None
    messages = matching["Message"].fillna("").astype(str)
    crash = matching[
        messages.str.contains(r"crash|collision|accident|impact", case=False, regex=True)
        & ~messages.str.contains("pit", case=False, regex=False)
    ]
    if not crash.empty:
        message = crash.sort_values("seconds").iloc[0]
        stopped_at = safe_number(message.get("seconds"))
        if stopped_at is None:
            driver_positions = locations[locations["driver_number"] == int(driver_number)]
            stopped_at = safe_number(driver_positions["seconds"].max()) if not driver_positions.empty else None
        if stopped_at is not None:
            return {
                "time": stopped_at,
                "lap": safe_number(message.get("Lap")),
                "reason": str(message.get("Message", "Crash reported")),
                "status": "Crash reported",
            }

    # Some sessions report the incident without a driver number. If the car
    # enters the pits and never has another pit exit, treat that final stop as
    # a retirement; ordinary pit stops have a later PitOutTime and stay shown.
    if not driver_laps.empty and {"PitInTime_seconds", "PitOutTime_seconds"}.issubset(driver_laps.columns):
        pit_in = driver_laps["PitInTime_seconds"].dropna()
        pit_out = driver_laps["PitOutTime_seconds"].dropna()
        if not pit_in.empty:
            final_pit_in = safe_number(pit_in.max())
            final_pit_out = safe_number(pit_out.max()) if not pit_out.empty else None
            if final_pit_in is not None and (final_pit_out is None or final_pit_out < final_pit_in):
                return {
                    "time": final_pit_in,
                    "lap": None,
                    "reason": "Returned to the pits and did not rejoin",
                    "status": "Retired",
                }
    return None


browser_drivers = []
for _, driver in drivers.iterrows():
    number = int(driver["driver_number"])
    samples = locations[locations["driver_number"] == number].sort_values("seconds").copy()
    samples = samples[samples["seconds"] >= replay_start]
    # Preserve roughly the source telemetry cadence; requestAnimationFrame
    # interpolates between these samples for a smooth 60fps display.
    samples["bucket"] = (samples["seconds"] * 4).astype(int)
    samples = samples.drop_duplicates("bucket", keep="last")

    lap_records = []
    if not laps.empty:
        driver_laps = laps[laps["driver_number"] == number].sort_values("seconds")
        for _, lap in driver_laps.iterrows():
            if pd.isna(lap.get("seconds")):
                continue
            lap_records.append({
                "t": safe_number(lap.get("seconds")),
                "lapStart": safe_number(lap.get("lap_start_seconds")),
                "lap": safe_number(lap.get("LapNumber")),
                "pos": safe_number(lap.get("Position")),
                "lapTime": safe_number(lap.get("lap_seconds")),
                "valid": (pd.isna(lap.get("IsAccurate")) or bool(lap.get("IsAccurate")))
                and (pd.isna(lap.get("Deleted")) or not bool(lap.get("Deleted"))),
                "sectors": [safe_number(lap.get(f"sector{i}_seconds")) for i in (1, 2, 3)],
                "tyre": None if pd.isna(lap.get("Compound")) else str(lap.get("Compound")),
                "tyreAge": safe_number(lap.get("TyreLife")),
            })

    def clean_text(value):
        return "" if pd.isna(value) else str(value)

    driver_laps = laps[laps["driver_number"] == number] if not laps.empty else pd.DataFrame()
    driver_telemetry = (
        car_telemetry[car_telemetry["driver_number"] == number].copy()
        if not car_telemetry.empty else pd.DataFrame()
    )
    telemetry_columns = ["Speed", "Throttle", "Brake", "RPM", "nGear", "DRS", "active_aero"]
    telemetry_samples = []
    if not driver_telemetry.empty:
        for _, sample in driver_telemetry.iterrows():
            telemetry_samples.append([
                safe_number(sample.get("seconds")),
                *[safe_number(sample.get(column)) for column in telemetry_columns],
            ])
    dnf = (
        dnf_info(number, driver.get("result_status", ""), driver_laps)
        if session_code == "R"
        else crash_info(number, driver_laps)
    )
    browser_drivers.append({
        "number": number,
        "abbr": clean_text(driver["name_acronym"]),
        "name": clean_text(driver["full_name"]),
        "team": clean_text(driver["team_name"]),
        "color": "#" + clean_text(driver["team_colour"]).lstrip("#"),
        "photo": clean_text(driver.get("headshot_url", "")),
        "grid": safe_number(driver["grid_position"]),
        "dnf": dnf,
        "samples": samples[["seconds", "X", "Y"]].round(2).values.tolist(),
        "laps": lap_records,
        "telemetry": telemetry_samples,
    })


def browser_driver_order(driver):
    dnf = driver["dnf"]
    if dnf is None:
        return (0, 0, 0, 0)
    status = str(dnf.get("status", "")).strip().lower()
    is_dns = status == "dns" or "did not start" in status
    if is_dns:
        return (1, 1, safe_number(driver["grid"]) or 999, 0)
    lap = dnf.get("lap") if dnf.get("lap") is not None else -1
    return (1, 0, -lap, dnf.get("time") or 0)


browser_drivers.sort(key=browser_driver_order)

TEAM_LOGO_FILES = {
    "audi": "audi.svg",
    "cadillac": "cadillac.svg",
    "alpine": "alpine.svg",
    "mclaren": "mclaren.svg",
    "ferrari": "ferrari.svg",
    "mercedes": "mercedes.svg",
    "haas": "haas.svg",
    "racing-bulls": "racing-bulls.svg",
    "williams": "williams.svg",
    "aston-martin": "aston-martin.svg",
    "sauber": "sauber.svg",
    "alfa-romeo": "alfa-romeo.svg",
    "force-india": "force-india.svg",
}


def logo_key_for_team(team):
    name = str(team or "").lower()
    if "audi" in name:
        return "audi"
    if "cadillac" in name:
        return "cadillac"
    if "alpine" in name:
        return "alpine"
    if "mclaren" in name:
        return "mclaren"
    if "ferrari" in name:
        return "ferrari"
    if "mercedes" in name:
        return "mercedes"
    if "haas" in name:
        return "haas"
    if "racing bulls" in name or "alphatauri" in name or "toro rosso" in name:
        return "racing-bulls"
    if "williams" in name:
        return "williams"
    if "aston martin" in name:
        return "aston-martin"
    if "sauber" in name or "kick" in name:
        return "sauber"
    if "alfa romeo" in name:
        return "alfa-romeo"
    if "force india" in name or "racing point" in name:
        return "force-india"
    return None


for browser_driver in browser_drivers:
    browser_driver["logoKey"] = logo_key_for_team(browser_driver.get("team"))


team_logo_data = {
    key: "data:image/svg+xml;base64," + base64.b64encode(
        (Path(__file__).resolve().parent / "assets" / "team-logos" / filename).read_bytes()
    ).decode("ascii")
    for key, filename in TEAM_LOGO_FILES.items()
    if (Path(__file__).resolve().parent / "assets" / "team-logos" / filename).is_file()
}

official_dotd = None
if session_code == "R":
    try:
        official_dotd = get_official_dotd(year, event["RoundNumber"])
    except Exception:
        # DOTD is supplemental; an unavailable award source must not block replay.
        official_dotd = None

payload = {
    "track": track[["X", "Y"]].round(2).values.tolist(),
    "drivers": browser_drivers,
    "teamLogos": team_logo_data,
    "trackStatus": (
        track_status[["seconds", "Status", "Message"]].dropna(subset=["seconds"])
        .sort_values("seconds").to_dict("records")
        if not track_status.empty else []
    ),
    "start": replay_start,
    "duration": replay_duration,
    "event": str(event["EventName"]),
    "session": session_name,
    "year": int(year),
    "totalLaps": total_laps,
    "officialDotd": official_dotd,
}
data_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")


player_html = r"""<!doctype html>
<html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;background:#0b1018;color:#e8edf5;font:14px system-ui,-apple-system,sans-serif}
.shell{padding:12px 16px 16px}.head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;color:#9aa4b2}
.head-left{display:flex;align-items:center;gap:12px;min-width:0}.view-control{display:flex;align-items:center;gap:7px;color:#929dad;font-size:12px}.view-control select{padding:6px 9px;font-size:12px}
.driver-picker{min-width:200px;background:#1b2330;color:#f4f6fa;border:1px solid #343e4d;border-radius:8px;padding:8px 12px;font:inherit}
.race-layout{display:grid;grid-template-columns:minmax(0,1fr) 430px;gap:12px;align-items:stretch}
.highlights-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;margin-top:10px}.highlight-card{background:#151d28;border:1px solid #283241;border-radius:9px;padding:9px 11px;min-width:0}.highlight-label{color:#8793a3;font-size:9px;letter-spacing:.06em;text-transform:uppercase}.highlight-value{margin-top:5px;color:#eef2f7;font-size:12px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.highlight-sub{margin-top:3px;color:#aab6c5;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.purple{color:#c084fc!important}.green{color:#48d597!important}
.map-pane{min-width:0}canvas{display:block;width:100%;height:540px;background:#0b1018;border-radius:12px}
.pov-toolbar{display:none;align-items:center;gap:10px;margin:0 0 10px;color:#9aa4b2;font-size:12px}.pov-toolbar .driver-picker{max-width:320px}.pov-note{margin-left:auto;color:#8793a3;font-size:11px;text-align:right}
#pov-canvas{display:none}
body.pov-view #map{display:none}body.pov-view #pov-canvas{display:block;height:540px;width:100%;background:#0b1018;border-radius:12px}body.pov-view .pov-toolbar{display:flex}
.right-rail{display:flex;flex-direction:column;min-height:540px;max-height:540px;overflow:hidden}
.panel{background:#151d28;border:1px solid #283241;border-radius:10px;padding:12px;min-height:0}
.panel h3{margin:0 0 10px;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:#aeb8c7}
.tower-title{display:flex;align-items:center;justify-content:space-between;gap:10px}.tower-title h3{margin:0 0 10px}.driver-search{width:min(48%,220px);margin:0 0 10px;background:#0f1620;color:#eef2f7;border:1px solid #344154;border-radius:7px;padding:7px 9px;font:12px system-ui}.driver-search::placeholder{color:#8290a2}
.tower-panel{flex:1;overflow:auto}.tower-head,.tower-row{display:grid;grid-template-columns:24px 38px 42px 54px 66px 58px minmax(78px,1fr);gap:5px;align-items:center;min-width:400px}.tower-head{padding:0 0 7px;color:#8290a2;font-size:9px;letter-spacing:.04em;text-transform:uppercase}.tower-row{padding:7px 0;border-top:1px solid #27303d;font-size:11px;font-variant-numeric:tabular-nums;cursor:pointer}.tower-row:hover{background:#202b39}.tower-row .pos{color:#929dad}.tower-row .abbr{font-weight:700}.tower-row .current-lap,.tower-row .gap,.tower-row .lap-info,.tower-row .delta,.tower-row .tyre{color:#929dad;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.tower-row.dnf-entry{opacity:.68}.tower-row.dnf-entry .pos,.tower-row.dnf-entry .gap{color:#ff7b83}.empty{font-size:12px;color:#929dad}
.tower-row.selected-driver{background:#351a22;box-shadow:inset 3px 0 #ef3340}.tower-row.selected-driver span{color:#ff6570!important}.tower-row.selected-driver .abbr{color:#ff4655!important}
body.extended-timing .race-layout{grid-template-columns:minmax(0,1fr) 580px}
body.extended-timing .map-pane canvas{height:680px}
body.extended-timing.pov-view #pov-canvas{height:680px}
body.extended-timing .right-rail{min-height:720px;max-height:720px}
body.extended-timing .tower-head,body.extended-timing .tower-row{grid-template-columns:30px 42px 82px 68px 100px 72px 84px;gap:7px;min-width:0}
body.extended-timing .tower-row{font-size:12px;padding:8px 0}
@media(max-width:850px){.race-layout{grid-template-columns:1fr}.right-rail{min-height:300px;max-height:420px}}
@media(max-width:850px){.highlights-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:1100px){body.extended-timing .race-layout{grid-template-columns:minmax(0,1fr) 520px}}
@media(max-width:850px){body.extended-timing .race-layout{grid-template-columns:1fr}body.extended-timing .map-pane canvas{height:540px}body.extended-timing .right-rail{min-height:720px;max-height:720px}}
.controls{display:flex;align-items:center;gap:12px;margin-top:12px}.btn,select{background:#1b2330;color:#f4f6fa;border:1px solid #343e4d;border-radius:8px;padding:8px 12px;font:inherit}
input[type=range]{flex:1;accent-color:#ef3340}.time{min-width:115px;text-align:right;color:#cbd2dc;font-variant-numeric:tabular-nums}
.driver-detail{display:none;margin-top:12px;padding:14px;background:#151d28;border:1px solid #283241;border-radius:10px;color:#d7dee8;grid-template-columns:92px minmax(0,1fr);gap:14px;position:relative}
.driver-detail.open{display:grid}.portrait{width:92px;height:110px;border-radius:9px;background:#273244;display:grid;place-items:center;overflow:hidden}.portrait img{width:100%;height:100%;object-fit:cover}.portrait-fallback{font-size:22px;font-weight:800;color:#d9e2ef}
.driver-copy{min-width:0}.driver-identity{display:flex;align-items:center;gap:10px;margin:0 32px 10px 0}.driver-abbr{font-size:18px;font-weight:800}.driver-name{font-size:19px;font-weight:750;color:#fff}.driver-team{font-size:12px;color:#929dad;margin-top:2px}.driver-state{font-size:11px;margin-left:auto;padding:4px 7px;background:#202b39;border-radius:6px;color:#aab6c5}.close-detail{position:absolute;right:10px;top:8px;border:0;background:transparent;color:#aab6c5;font-size:20px;cursor:pointer}.driver-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:7px}.metric{background:#1b2431;border-radius:7px;padding:7px 9px;min-width:0}.metric-label{font-size:9px;letter-spacing:.05em;text-transform:uppercase;color:#8793a3}.metric-value{margin-top:3px;font-size:13px;font-weight:650;color:#eef2f7;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.muted{color:#929dad}.tip{font-size:12px;color:#8e99a8;margin:8px 0 0}
.telemetry-panel{display:none;flex:1;min-height:0;margin:0;padding:12px;background:#111923;border:1px solid #283241;border-radius:10px;overflow-y:auto}.telemetry-title{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;margin-bottom:12px}.telemetry-title h3{margin:0;color:#eef2f7;font-size:14px}.telemetry-title span{color:#929dad;font-size:11px}.telemetry-metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;margin-bottom:10px}.telemetry-stack{display:flex;flex-direction:column;gap:8px;padding-right:3px}.telemetry-chart{display:block;width:100%;height:150px;background:#0b1018;border-radius:8px;flex:0 0 150px}.telemetry-chart[hidden]{display:none!important}.telemetry-empty{padding:20px;text-align:center;color:#929dad;background:#151d28;border-radius:10px}
body.telemetry-view .tower-panel{display:none}body.telemetry-view .telemetry-panel{display:block}
body.telemetry-view .right-rail{overflow-y:auto}
body.combined-view .tower-panel{display:block}
body.combined-view .telemetry-panel{display:block;flex:none;min-height:auto;margin-top:12px;overflow:visible}
</style></head><body><div class="shell">
<div class="head"><div class="head-left"><strong id="heading"></strong><label class="view-control">View<select id="view-mode" aria-label="Choose replay view"><option value="timing">Map + timing tower</option><option value="telemetry">Map + telemetry</option><option value="combined">Map + timing + telemetry</option><option value="pov">Generated 3D driver view</option></select></label></div><span id="lap-count"></span><span id="flag-state">Track status · clear</span></div>
<div class="race-layout"><div class="map-pane"><div class="pov-toolbar"><label for="pov-driver-select">Driver</label><select id="pov-driver-select" class="driver-picker" aria-label="Choose driver for generated viewpoint"><option value="">Choose a driver</option></select><label for="pov-camera-select">Camera</label><select id="pov-camera-select" aria-label="Choose 3D camera"><option value="driver">Driver view</option><option value="overview">3D track overview</option></select><span class="pov-note">Generated from recorded track coordinates · elevation is not available</span></div><canvas id="map"></canvas><canvas id="pov-canvas" aria-label="Generated 3D driver viewpoint and track overview"></canvas></div>
<aside class="right-rail">
  <section class="panel tower-panel"><div class="tower-title"><h3>Timing tower</h3><input id="driver-search" class="driver-search" type="search" list="driver-search-options" placeholder="Find a driver…" aria-label="Find driver by name or abbreviation"><datalist id="driver-search-options"></datalist></div><div id="tower-header" class="tower-head"><span>Pos</span><span>Driver</span><span>Lap</span><span title="Estimated time to the car ahead">Gap</span><span>Last lap</span><span>Δ lap</span><span>Tyre</span></div><div id="timing-tower"></div></section>
</aside></div>
<div id="session-highlights" class="highlights-grid" aria-live="polite"></div>
<div class="controls"><button id="play" class="btn">▶ Play</button><button id="jump-start" class="btn" title="Return to lights out / session start">↺ Start</button>
<select id="speed"><option value="1">1×</option><option value="2">2×</option><option value="5" selected>5×</option><option value="10">10×</option><option value="20">20×</option></select>
<input id="seek" type="range" min="0" step="0.1"><span id="clock" class="time"></span></div>
<div id="detail" class="driver-detail" aria-live="polite">
  <div class="portrait"><img id="driver-photo" alt=""><span id="photo-fallback" class="portrait-fallback"></span></div>
  <div class="driver-copy"><button id="close-detail" class="close-detail" aria-label="Close driver details">×</button>
    <div class="driver-identity"><span id="driver-abbr" class="driver-abbr"></span><div><div id="driver-name" class="driver-name"></div><div id="driver-team" class="driver-team"></div></div><span id="driver-state" class="driver-state"></span></div>
    <div id="driver-stats" class="driver-stats"></div>
  </div>
</div>
<section class="telemetry-panel" aria-label="Driver telemetry">
  <div class="telemetry-title"><h3 id="telemetry-driver">Car telemetry</h3><select id="telemetry-driver-select" class="driver-picker" aria-label="Choose driver"></select><span id="telemetry-time">Select a driver to inspect recorded telemetry</span></div>
  <div id="telemetry-metrics" class="telemetry-metrics"></div><div id="telemetry-empty" class="telemetry-empty">Click a driver on the map or timing tower, or choose one above, to see their telemetry.</div>
  <div id="telemetry-stack" class="telemetry-stack" hidden>
    <canvas id="chart-speed" class="telemetry-chart" aria-label="Speed telemetry chart"></canvas>
    <canvas id="chart-throttle" class="telemetry-chart" aria-label="Throttle telemetry chart"></canvas>
    <canvas id="chart-brake" class="telemetry-chart" aria-label="Brake telemetry chart"></canvas>
    <canvas id="chart-rpm" class="telemetry-chart" aria-label="Engine RPM telemetry chart"></canvas>
    <canvas id="chart-gear" class="telemetry-chart" aria-label="Gear telemetry chart"></canvas>
    <canvas id="chart-drs" class="telemetry-chart" aria-label="DRS telemetry chart"></canvas>
    <canvas id="chart-aero" class="telemetry-chart" aria-label="Active aero telemetry chart" hidden></canvas>
    <div id="active-aero-unavailable" class="telemetry-empty" hidden>Active aero replaced DRS for 2026, but this session has no active-aero telemetry in the FastF1 data.</div>
  </div>
</section>
<p class="tip">Gaps are estimated to the car ahead from interpolated track position; P1 is 0.0. Car dots are interpolated between recorded samples.</p>
</div><script>
const data=__DATA__;
const teamLogoImages={};
for(const [key,src] of Object.entries(data.teamLogos||{})){
  const image=new Image();image.onload=()=>{teamLogoImages[key]=image;if(typeof current!=="undefined")drawPov(data.start+current);};image.src=src;
}
const povNoiseCanvas=document.createElement("canvas");povNoiseCanvas.width=128;povNoiseCanvas.height=128;
const povNoiseCtx=povNoiseCanvas.getContext("2d"),povNoiseImage=povNoiseCtx.createImageData(128,128);
for(let i=0;i<povNoiseImage.data.length;i+=4){const shade=30+Math.random()*90;povNoiseImage.data[i]=shade;povNoiseImage.data[i+1]=shade;povNoiseImage.data[i+2]=shade;povNoiseImage.data[i+3]=Math.random()<.28?38:0;}
povNoiseCtx.putImageData(povNoiseImage,0,0);
document.getElementById("heading").textContent=data.event+" · "+data.session;
const canvas=document.getElementById("map"),ctx=canvas.getContext("2d"),povCanvas=document.getElementById("pov-canvas"),povCtx=povCanvas.getContext("2d"),seek=document.getElementById("seek");
const povAsphaltPattern=povCtx.createPattern(povNoiseCanvas,"repeat");
const viewMode=document.getElementById("view-mode");
const telemetryDriverSelect=document.getElementById("telemetry-driver-select");
const povDriverSelect=document.getElementById("pov-driver-select");
const povCameraSelect=document.getElementById("pov-camera-select");
const telemetryPanel=document.querySelector(".telemetry-panel"),rightRail=document.querySelector(".right-rail");
const detailPanel=document.getElementById("detail");
const telemetryStack=document.getElementById("telemetry-stack"),telemetryCharts=[
  ["chart-speed",1,"Speed", "#49a6ff", "km/h",400],
  ["chart-throttle",2,"Throttle", "#32d583", "%",100],
  ["chart-brake",3,"Brake", "#ef5965", "",1],
  ["chart-rpm",4,"Engine RPM", "#bd8cff", "rpm",15000],
  ["chart-gear",5,"Gear", "#ffd166", "",8],
  ["chart-drs",6,"DRS", "#4dd0e1", "",14],
  ["chart-aero",7,"Active aero", "#4dd0e1", "mode",1],
].map(([id,index,label,color,unit,max])=>{const canvas=document.getElementById(id);return {canvas,ctx:canvas.getContext("2d"),index,label,color,unit,max};});
const playButton=document.getElementById("play"),jumpStartButton=document.getElementById("jump-start"),speedSelect=document.getElementById("speed");
const clock=document.getElementById("clock"),detail=document.getElementById("detail");
const flagState=document.getElementById("flag-state");
const lapCount=document.getElementById("lap-count");
const timingTower=document.getElementById("timing-tower");
const towerHeader=document.getElementById("tower-header");
const driverSearch=document.getElementById("driver-search"),driverSearchOptions=document.getElementById("driver-search-options");
const driverPhoto=document.getElementById("driver-photo"),photoFallback=document.getElementById("photo-fallback");
const driverAbbr=document.getElementById("driver-abbr"),driverName=document.getElementById("driver-name");
const driverTeam=document.getElementById("driver-team"),driverState=document.getElementById("driver-state");
const driverStats=document.getElementById("driver-stats");
const telemetryDriver=document.getElementById("telemetry-driver"),telemetryTime=document.getElementById("telemetry-time");
const telemetryMetrics=document.getElementById("telemetry-metrics"),telemetryEmpty=document.getElementById("telemetry-empty");
seek.max=data.duration;
let current=0,playing=false,last=0,selected=null,boundsCache=null,lastTowerSecond=-1,lastDetailSecond=-1,lastPhotoDriver=null;
let lastTelemetrySecond=-1;
let lastPovRenderMs=0;
let sessionSectorBests=[null,null,null];
let povHeadingState=null,povHeadingTime=null;
telemetryDriverSelect.innerHTML='<option value="">Choose a driver</option>'+data.drivers.map(d=>`<option value="${d.number}">${escapeHtml(d.abbr)} · ${escapeHtml(d.name)}</option>`).join("");
povDriverSelect.innerHTML='<option value="">Choose a driver</option>'+data.drivers.map(d=>`<option value="${d.number}">${escapeHtml(d.abbr)} · ${escapeHtml(d.name)}</option>`).join("");
driverSearchOptions.innerHTML=data.drivers.flatMap(d=>[
  `<option value="${escapeHtml(d.name)}"></option>`,`<option value="${escapeHtml(d.abbr)}"></option>`
]).join("");
const isRace=data.session==="Race";
if(!isRace){document.body.classList.add("extended-timing");towerHeader.innerHTML="<span>Pos</span><span>Driver</span><span>Best lap</span><span>Gap</span><span>Last lap</span><span>Δ lap</span><span>Tyre</span>";}
viewMode.onchange=()=>{
  document.body.classList.toggle("telemetry-view",viewMode.value==="telemetry");
  document.body.classList.toggle("combined-view",viewMode.value==="combined");
  document.body.classList.toggle("pov-view",viewMode.value==="pov");
  if(viewMode.value==="telemetry")rightRail.appendChild(telemetryPanel);
  else detailPanel.insertAdjacentElement("afterend",telemetryPanel);
  requestAnimationFrame(()=>{resize();drawPov(data.start+current);});
};
function resize(){
  const dpr=window.devicePixelRatio||1,rect=canvas.getBoundingClientRect();
  canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);boundsCache=null;
  const povRect=povCanvas.getBoundingClientRect();
  if(povRect.width&&povRect.height){povCanvas.width=Math.round(povRect.width*dpr);povCanvas.height=Math.round(povRect.height*dpr);povCtx.setTransform(dpr,0,0,dpr,0,0);}
  draw();renderTelemetry(data.start+current,true);
  drawPov(data.start+current);
}
function bounds(){
  if(data.track.length<2)return null;
  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  for(const p of data.track){minX=Math.min(minX,p[0]);maxX=Math.max(maxX,p[0]);minY=Math.min(minY,p[1]);maxY=Math.max(maxY,p[1]);}
  const w=canvas.clientWidth,h=canvas.clientHeight,pad=34;
  const scale=Math.min((w-2*pad)/(maxX-minX||1),(h-2*pad)/(maxY-minY||1));
  return {minX,minY,maxY,scale,w,h,pad};
}
function xy(x,y,b){return [b.pad+(x-b.minX)*b.scale,b.h-b.pad-(y-b.minY)*b.scale];}
function positionAt(d,t){
  const a=d.samples;if(!a.length||t<a[0][0]||t>a[a.length-1][0])return null;
  let lo=0,hi=a.length-1;
  while(lo<hi){const m=(lo+hi)>>1;if(a[m][0]<t)lo=m+1;else hi=m;}
  const j=lo;if(j===0||a[j][0]===t)return [a[j][1],a[j][2]];
  const p=a[j-1],q=a[j],f=(t-p[0])/(q[0]-p[0]||1);
  return [p[1]+(q[1]-p[1])*f,p[2]+(q[2]-p[2])*f];
}
function smoothPositionAt(d,t){
  const samples=d.samples;if(samples.length<3)return positionAt(d,t);
  if(t<samples[0][0]||t>samples[samples.length-1][0])return positionAt(d,t);
  let lo=0,hi=samples.length-1;
  while(lo<hi){const mid=(lo+hi)>>1;if(samples[mid][0]<t)lo=mid+1;else hi=mid;}
  const j=lo;if(j===0||samples[j][0]===t)return [samples[j][1],samples[j][2]];
  const p=samples[j-1],q=samples[j],before=samples[Math.max(0,j-2)],after=samples[Math.min(samples.length-1,j+1)];
  const dt=q[0]-p[0]||1,u=(t-p[0])/dt,u2=u*u,u3=u2*u;
  const h00=2*u3-3*u2+1,h10=u3-2*u2+u,h01=-2*u3+3*u2,h11=u3-u2;
  return [1,2].map(axis=>{
    const m0=(q[axis]-before[axis])/(q[0]-before[0]||dt)*dt;
    const m1=(after[axis]-p[axis])/(after[0]-p[0]||dt)*dt;
    const value=h00*p[axis]+h10*m0+h01*q[axis]+h11*m1;
    const allowance=Math.hypot(q[1]-p[1],q[2]-p[2])*.08;
    return Math.max(Math.min(p[axis],q[axis])-allowance,Math.min(Math.max(p[axis],q[axis])+allowance,value));
  });
}
function drawSelectedPurpleSectors(b){
  if(selected===null)return;
  for(let index=0;index<3;index++){
    const best=sessionSectorBests[index];
    if(!best||best.driver.number!==selected)continue;
    const sectors=best.lap.sectors||[];
    if(sectors.slice(0,index+1).some(value=>value===null||value===undefined||value<=0))continue;
    const lapStart=best.lap.lapStart!==null?best.lap.lapStart:best.lap.t-best.lap.lapTime;
    if(!Number.isFinite(lapStart))continue;
    const startTime=lapStart+sectors.slice(0,index).reduce((sum,value)=>sum+value,0);
    const endTime=startTime+sectors[index];
    const points=[];
    const startPosition=positionAt(best.driver,startTime);
    if(startPosition)points.push(startPosition);
    for(const sample of best.driver.samples){if(sample[0]>startTime&&sample[0]<endTime)points.push([sample[1],sample[2]]);}
    const endPosition=positionAt(best.driver,endTime);
    if(endPosition)points.push(endPosition);
    if(points.length<2)continue;
    ctx.beginPath();
    points.forEach((point,pointIndex)=>{const q=xy(point[0],point[1],b);pointIndex?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});
    ctx.strokeStyle="#c084fc";ctx.lineWidth=8;ctx.lineCap="round";ctx.lineJoin="round";
    ctx.shadowColor="#c084fc";ctx.shadowBlur=14;ctx.stroke();ctx.shadowBlur=0;
  }
}
function lapAt(d,t){
  let result=null;
  for(const lap of d.laps){if(lap.t!==null&&lap.t<=t)result=lap;else if(lap.t!==null&&lap.t>t)break;}
  return result;
}
function lapDeltaAt(d,t){
  const completed=d.laps.filter(l=>l.t!==null&&l.t<=t&&l.lapTime!==null&&l.lapTime>0&&l.valid!==false);
  if(completed.length<2)return "—";
  const delta=completed[completed.length-1].lapTime-completed[completed.length-2].lapTime;
  return `${delta>0?"+":delta<0?"−":""}${Math.abs(delta).toFixed(3)}s`;
}
function lapNumberAt(d,t){
  const lap=lapAt(d,t);
  if(d.dnf&&d.dnf.time!==null&&t>=d.dnf.time){
    if(/did not start|\bDNS\b/i.test(d.dnf.status||""))return "DNS";
    return d.dnf.lap??(lap&&lap.lap!==null?lap.lap:"—");
  }
  const current=(lap&&lap.lap!==null?lap.lap:0)+1;
  return data.totalLaps?Math.min(data.totalLaps,current):current;
}
function trackStyleAt(t){
  let code="1";
  for(const item of data.trackStatus){if(item.seconds<=t)code=String(item.Status);else break;}
  if(code==="2")return {color:"#ffd43b",label:"Yellow flag"};
  if(code==="4")return {color:"#ffd43b",label:"Safety car"};
  if(code==="5")return {color:"#ff3b45",label:"Red flag"};
  if(code==="6"||code==="7")return {color:"#ff922b",label:"Virtual safety car"};
  return {color:"#596272",label:"Track clear"};
}
function escapeHtml(value){return String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));}
const trackSegments=[];let trackLength=0;
for(let i=0;i<data.track.length-1;i++){
  const a=data.track[i],b=data.track[i+1],dx=b[0]-a[0],dy=b[1]-a[1],length=Math.hypot(dx,dy);
  if(length>0){trackSegments.push({x:a[0],y:a[1],dx,dy,length,start:trackLength});trackLength+=length;}
}
function trackProgress(position){
  if(!position||!trackSegments.length||!trackLength)return 0;
  let bestDistance=Infinity,bestDistanceAlong=0;
  for(const s of trackSegments){
    const f=Math.max(0,Math.min(1,((position[0]-s.x)*s.dx+(position[1]-s.y)*s.dy)/(s.length*s.length)));
    const px=s.x+f*s.dx,py=s.y+f*s.dy,distance=(position[0]-px)**2+(position[1]-py)**2;
    if(distance<bestDistance){bestDistance=distance;bestDistanceAlong=s.start+f*s.length;}
  }
  return bestDistanceAlong/trackLength;
}
function trackProjection(position){
  if(!position||!trackSegments.length)return null;
  let bestDistance=Infinity,best=null;
  for(const s of trackSegments){
    const f=Math.max(0,Math.min(1,((position[0]-s.x)*s.dx+(position[1]-s.y)*s.dy)/(s.length*s.length)));
    const x=s.x+f*s.dx,y=s.y+f*s.dy,distance=(position[0]-x)**2+(position[1]-y)**2;
    if(distance<bestDistance){bestDistance=distance;best={distanceAlong:s.start+f*s.length,x,y,tx:s.dx/s.length,ty:s.dy/s.length};}
  }
  return best;
}
function trackPointAt(distanceAlong){
  if(!trackSegments.length||!trackLength)return null;
  let distance=((distanceAlong%trackLength)+trackLength)%trackLength;
  let s=trackSegments[trackSegments.length-1];
  for(const segment of trackSegments){if(distance<=segment.start+segment.length){s=segment;break;}}
  const f=Math.max(0,Math.min(1,(distance-s.start)/s.length));
  return {x:s.x+f*s.dx,y:s.y+f*s.dy,tx:s.dx/s.length,ty:s.dy/s.length};
}
function drawTeamMark(g,x,y,team,color,logoKey=null){
  const name=String(team||"").toLowerCase();
  g.save();g.translate(x,y);g.fillStyle="#101722";g.strokeStyle=color;g.lineWidth=2;g.beginPath();g.roundRect(0,0,48,42,8);g.fill();g.stroke();
  const suppliedLogo=logoKey&&teamLogoImages[logoKey];
  if(suppliedLogo){
    g.fillStyle="#f5f6f7";g.beginPath();g.roundRect(3,3,42,36,5);g.fill();
    const ratio=Math.min(38/suppliedLogo.width,31/suppliedLogo.height),logoW=suppliedLogo.width*ratio,logoH=suppliedLogo.height*ratio;
    g.drawImage(suppliedLogo,24-logoW/2,21-logoH/2,logoW,logoH);g.restore();return;
  }
  g.textAlign="center";g.textBaseline="middle";
  if(name.includes("ferrari")){
    g.fillStyle="#f3c843";g.beginPath();g.moveTo(12,5);g.lineTo(36,5);g.lineTo(33,31);g.lineTo(24,38);g.lineTo(15,31);g.closePath();g.fill();g.fillStyle="#b51221";g.font="900 13px system-ui";g.fillText("SF",24,22);
  }else if(name.includes("red bull")){
    g.fillStyle="#193c73";g.beginPath();g.ellipse(24,21,19,14,0,0,Math.PI*2);g.fill();g.fillStyle="#f6c644";g.beginPath();g.arc(14,18,6,0,Math.PI*2);g.fill();g.fillStyle="#fff";g.font="900 11px system-ui";g.fillText("RB",29,23);
  }else if(name.includes("mercedes")){
    g.strokeStyle="#dce5e8";g.lineWidth=2;g.beginPath();g.arc(24,21,15,0,Math.PI*2);g.stroke();for(let a=-Math.PI/2;a<Math.PI*1.6;a+=Math.PI*2/3){g.beginPath();g.moveTo(24,21);g.lineTo(24+14*Math.cos(a),21+14*Math.sin(a));g.stroke();}
  }else if(name.includes("aston martin")){
    g.strokeStyle="#b6d5c7";g.lineWidth=2;g.beginPath();g.moveTo(7,17);g.quadraticCurveTo(24,7,41,17);g.lineTo(34,16);g.lineTo(24,20);g.lineTo(14,16);g.closePath();g.stroke();g.fillStyle="#dce9e2";g.font="800 10px system-ui";g.fillText("AM",24,29);
  }else{
    let mark="";
    if(name.includes("mclaren"))mark="M";else if(name.includes("racing bulls"))mark="RB";else if(name.includes("williams"))mark="W";else if(name.includes("alpine"))mark="A";else if(name.includes("haas"))mark="H";else if(name.includes("cadillac"))mark="C";else mark=String(team||"F1").replace(/[^A-Za-z]/g,"").slice(0,3).toUpperCase();
    g.fillStyle=color;g.font="900 20px system-ui";g.fillText(mark,24,22);
  }
  g.restore();
}
function drawTrack3D(t,w,h){
  const g=povCtx,source=data.track;
  if(!source||source.length<3){g.fillStyle="#d7dee8";g.font="14px system-ui";g.textAlign="center";g.fillText("A 3D circuit model is unavailable for this session",w/2,h/2);return;}
  // Keep the map's X/Y orientation and route order. A shallow camera tilt
  // adds depth without rotating or reshaping the circuit between views.
  const pitch=.18,sp=Math.sin(pitch),cp=Math.cos(pitch);
  const xs=source.map(p=>p[0]),ys=source.map(p=>p[1]);
  const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
  const spanX=Math.max(1,maxX-minX),spanY=Math.max(1,maxY-minY);
  const scale=Math.min((w-68)/spanX,(h-68)/spanY),cx=(minX+maxX)/2,cy=(minY+maxY)/2;
  const originX=w/2,originY=h*.54;
  function project(x,y,z=0){return [originX+(x-cx)*scale,originY-(y-cy)*scale*cp-z*scale*sp];}
  const half=58,segments=[];
  for(let i=0;i<source.length-1;i++){
    const a=source[i],b=source[i+1],dx=b[0]-a[0],dy=b[1]-a[1],len=Math.hypot(dx,dy);if(!len)continue;
    const nx=dy/len,ny=-dx/len;
    const leftA=project(a[0]+nx*half,a[1]+ny*half),rightA=project(a[0]-nx*half,a[1]-ny*half);
    const leftB=project(b[0]+nx*half,b[1]+ny*half),rightB=project(b[0]-nx*half,b[1]-ny*half);
    const depth=(a[1]+b[1])/2;
    segments.push({i,depth,leftA,rightA,leftB,rightB,a,b,nx,ny});
  }
  // Draw in the same source order as the 2D map so crossing sections retain
  // the same route continuity and the camera does not invent a different path.
  const road=trackStyleAt(t);
  for(const s of segments){
    const {leftA,rightA,leftB,rightB}=s;
    g.beginPath();g.moveTo(...leftA);g.lineTo(...rightA);g.lineTo(...rightB);g.lineTo(...leftB);g.closePath();
    g.fillStyle="#313a45";g.fill();
    g.strokeStyle="#687487";g.lineWidth=Math.max(1.4,Math.min(4,scale*.016));g.stroke();
    if(s.i%6<3){
      for(const side of [-1,1]){
        const q0=side<0?leftA:rightA,q1=side<0?leftB:rightB;
        g.beginPath();g.moveTo(...q0);g.lineTo(...q1);g.strokeStyle=s.i%2===0?"#e8edf5":"#d94c57";g.lineWidth=Math.max(2,Math.min(8,scale*.045));g.stroke();
      }
    }
  }
  // Rails sit just outside the track edge, with posts that make the road ribbon read as a 3D object.
  for(const side of [-1,1]){
    let rail=[];
    for(let i=0;i<source.length;i+=4){
      const a=source[i],before=source[Math.max(0,i-1)],after=source[Math.min(source.length-1,i+1)];
      const dx=after[0]-before[0],dy=after[1]-before[1],len=Math.hypot(dx,dy)||1,nx=dy/len,ny=-dx/len;
      rail.push(project(a[0]+nx*side*115,a[1]+ny*side*115,30));
    }
    g.beginPath();rail.forEach((p,i)=>i?g.lineTo(...p):g.moveTo(...p));g.strokeStyle="#252f3a";g.lineWidth=Math.max(2,Math.min(9,scale*.06));g.stroke();
    g.beginPath();rail.forEach((p,i)=>i?g.lineTo(...p):g.moveTo(...p));g.strokeStyle="rgba(190,204,220,.55)";g.lineWidth=Math.max(1,Math.min(2,scale*.014));g.stroke();
  }
  // Current driver positions are placed on the 3D road surface.
  for(const d of data.drivers){
    if(d.dnf&&d.dnf.time!==null&&t>=d.dnf.time)continue;
    const p=positionAt(d,t);if(!p)continue;const q=project(p[0],p[1],8),isSelected=d.number===selected;
    const loc=trackProjection(p),ahead=loc?project(p[0]+loc.tx*180,p[1]+loc.ty*180,8):null;
    const angle=ahead?Math.atan2(ahead[1]-q[1],ahead[0]-q[0]):0,carScale=isSelected?1.25:1;
    g.save();g.translate(q[0],q[1]);g.rotate(angle);g.scale(carScale,carScale);
    g.fillStyle="#080c12";g.fillRect(-7,-7,5,4);g.fillRect(2,-7,5,4);g.fillRect(-7,3,5,4);g.fillRect(2,3,5,4);
    g.beginPath();g.moveTo(12,0);g.lineTo(6,-4);g.lineTo(-5,-3.5);g.lineTo(-7,0);g.lineTo(-5,3.5);g.lineTo(6,4);g.closePath();g.fillStyle=d.color||"#d7dee8";g.fill();
    g.strokeStyle=isSelected?"#fff":"rgba(8,12,18,.85)";g.lineWidth=isSelected?2:1;g.stroke();g.restore();
    if(isSelected||w>850){g.font=isSelected?"800 11px system-ui":"700 9px system-ui";g.textAlign="center";g.textBaseline="bottom";g.fillStyle="#f4f6fa";g.shadowColor="#0b1018";g.shadowBlur=3;g.fillText(d.abbr,q[0],q[1]-9);g.shadowBlur=0;}
  }
  const driver=data.drivers.find(d=>d.number===selected),flag=road.label;
  g.fillStyle="rgba(8,13,20,.9)";g.beginPath();g.roundRect(16,16,Math.min(w-32,370),74,12);g.fill();g.strokeStyle="rgba(178,192,209,.3)";g.stroke();
  if(driver){drawTeamMark(g,28,31,driver.team,driver.color||"#8b9bad",driver.logoKey);g.textAlign="left";g.textBaseline="top";g.fillStyle="#f5f7fa";g.font="800 16px system-ui";g.fillText(driver.name,88,30,260);g.fillStyle=driver.color||"#aab6c5";g.font="600 12px system-ui";g.fillText(driver.team||"Team",88,53,260);}
  else{g.textAlign="left";g.textBaseline="top";g.fillStyle="#f5f7fa";g.font="800 16px system-ui";g.fillText("3D circuit overview",28,28);g.fillStyle="#aab6c5";g.font="12px system-ui";g.fillText("Select a driver to highlight their car",28,53);}
  g.textAlign="right";g.textBaseline="top";g.fillStyle=road.color;g.font="700 11px system-ui";g.fillText(`${data.event} · ${flag}`,w-18,20);
  g.fillStyle="#aab6c5";g.font="10px system-ui";g.fillText("3D road mesh · elevation unavailable",w-18,38);
}
function drawPovEnvironment(g,w,h,horizon){
  // Trackside architecture frames the road like a broadcast onboard view.
  // It is deliberately generic because the source data has no 3D venue assets.
  const stand=g.createLinearGradient(0,h*.22,0,h*.52);stand.addColorStop(0,"#252638");stand.addColorStop(.52,"#45445a");stand.addColorStop(1,"#20232c");
  g.beginPath();g.moveTo(0,h*.2);g.lineTo(w*.29,h*.27);g.lineTo(w*.43,h*.48);g.lineTo(0,h*.49);g.closePath();g.fillStyle=stand;g.fill();
  g.beginPath();g.moveTo(0,h*.19);g.lineTo(w*.3,h*.26);g.lineTo(w*.35,h*.29);g.lineTo(0,h*.25);g.closePath();g.fillStyle="#161a23";g.fill();
  // Repeating seat rows and small crowd lights give the grandstand visual scale.
  for(let row=0;row<7;row++){
    const y0=h*(.29+row*.023),xEnd=w*(.305+row*.018),yEnd=h*(.30+row*.026);
    g.beginPath();g.moveTo(0,y0);g.lineTo(xEnd,yEnd);g.strokeStyle=row%2?"rgba(193,91,84,.48)":"rgba(216,220,227,.32)";g.lineWidth=Math.max(1,h*.004);g.stroke();
    for(let n=0;n<26;n++){
      const fx=(n+.5)/26,fy=((n*13+row*7)%11)/11,x=fx*xEnd,y=y0+(yEnd-y0)*fx+fy*h*.018;
      g.fillStyle=["#d5d1c6","#c75a58","#597ca1","#343b4b"][((n+row*3)%4)];g.fillRect(x,y,Math.max(1,w*.0017),Math.max(1,h*.003));
    }
  }
  // Lit pit building on the right, with glass bays and garage shutters.
  const building=g.createLinearGradient(w*.64,h*.2,w,h*.48);building.addColorStop(0,"#414654");building.addColorStop(.45,"#77787c");building.addColorStop(1,"#292e39");
  g.beginPath();g.moveTo(w*.61,h*.47);g.lineTo(w*.69,h*.25);g.lineTo(w,h*.15);g.lineTo(w,h*.5);g.closePath();g.fillStyle=building;g.fill();
  g.beginPath();g.moveTo(w*.67,h*.25);g.lineTo(w,h*.145);g.lineTo(w,h*.205);g.lineTo(w*.65,h*.31);g.closePath();g.fillStyle="#202632";g.fill();
  for(let col=0;col<13;col++){
    const f=col/13,x=w*(.69+f*.34),top=h*(.275-f*.105),ww=w*.023,hh=h*.105;
    g.fillStyle=col%3===0?"#f4dcb0":"#b8cbd0";g.globalAlpha=col%3===0?.9:.72;g.fillRect(x,top,ww,hh);g.globalAlpha=1;
    g.strokeStyle="rgba(25,31,42,.72)";g.lineWidth=Math.max(1,w*.0016);g.strokeRect(x,top,ww,hh);
    g.fillStyle="#363b45";g.fillRect(x,top+hh+h*.035,ww,h*.085);
    g.strokeStyle="rgba(211,218,222,.3)";g.beginPath();g.moveTo(x,top+hh+h*.075);g.lineTo(x+ww,top+hh+h*.075);g.stroke();
  }
  // Floodlight poles and beams sit above the stands in the far field.
  for(const [x,y] of [[w*.33,h*.27],[w*.56,h*.24],[w*.82,h*.20]]){
    g.strokeStyle="rgba(30,36,46,.9)";g.lineWidth=Math.max(1,w*.002);g.beginPath();g.moveTo(x,y+h*.16);g.lineTo(x,y);g.stroke();
    const glow=g.createRadialGradient(x,y,1,x,y,h*.12);glow.addColorStop(0,"rgba(255,248,220,.28)");glow.addColorStop(1,"rgba(255,248,220,0)");g.fillStyle=glow;g.fillRect(x-h*.12,y-h*.12,h*.24,h*.24);
    g.fillStyle="#fff5d5";g.fillRect(x-w*.008,y-h*.004,w*.016,h*.008);
  }
}
function drawPovHud(t,driver,lap,sample,w,h){
  const g=povCtx,glass="rgba(5,9,15,.82)",line="rgba(223,231,241,.22)",muted="#b5bfcc",white="#f5f7fa";
  function panel(x,y,pw,ph){g.fillStyle=glass;g.beginPath();g.roundRect(x,y,pw,ph,7);g.fill();g.strokeStyle=line;g.lineWidth=1;g.stroke();}
  function label(text,x,y,size=10,color=muted,weight=700,align="left"){g.fillStyle=color;g.font=`${weight} ${size}px system-ui`;g.textAlign=align;g.textBaseline="middle";g.fillText(text,x,y);}
  const leftW=Math.max(158,Math.min(230,w*.205)),leftX=14,leftY=103,rowH=20;
  const running=data.drivers.filter(d=>!(d.dnf&&d.dnf.time!==null&&t>=d.dnf.time));
  const ranked=running.map(d=>{
    const completed=d.laps.filter(item=>item.t!==null&&item.t<=t&&item.lapTime!==null&&item.lapTime>0&&item.valid!==false);
    const last=lapAt(d,t),position=positionAt(d,t),progress=(last&&last.lap||0)+trackProgress(position);
    const best=completed.length?Math.min(...completed.map(item=>item.lapTime)):null;
    return {d,last,progress,best};
  }).sort((a,b)=>data.session==="Race"?b.progress-a.progress:(a.best??Infinity)-(b.best??Infinity));
  const shown=ranked.slice(0,Math.max(4,Math.min(7,Math.floor((h*.45-30)/rowH))));
  const boardH=26+shown.length*rowH;
  panel(leftX,leftY,leftW,boardH);label("POS   DRIVER",leftX+9,leftY+14,9,"#d3dbe5",800);label("LAST LAP",leftX+leftW-64,leftY+14,8,"#d3dbe5",800);
  g.strokeStyle=line;g.beginPath();g.moveTo(leftX+8,leftY+26);g.lineTo(leftX+leftW-8,leftY+26);g.stroke();
  shown.forEach((row,index)=>{
    const y=leftY+36+index*rowH;
    label(String(index+1).padStart(2," "),leftX+9,y,10,muted,600);
    label(row.d.abbr,leftX+42,y,11,row.d.color||white,800);
    const lapText=row.last&&row.last.lapTime?formatLapTime(row.last.lapTime):"—";
    label(lapText,leftX+leftW-9,y,9,white,600,"right");
    if(index<shown.length-1){g.strokeStyle="rgba(223,231,241,.1)";g.beginPath();g.moveTo(leftX+8,y+10);g.lineTo(leftX+leftW-8,y+10);g.stroke();}
  });
  const rightW=Math.max(250,Math.min(330,w*.29)),rightX=w-rightW-14,rightY=14;
  panel(rightX,rightY,rightW,124);
  label(`${data.event.toUpperCase()}  ·  ${data.session.toUpperCase()}`,rightX+12,rightY+15,9,"#d3dbe5",800);
  const lapNo=lap&&lap.lap?lap.lap:lapNumberAt(driver,t);
  label(`LAP ${lapNo}${data.totalLaps?" / "+data.totalLaps:""}`,rightX+12,rightY+40,17,white,800);
  const lapTime=lap&&lap.lapTime?formatLapTime(lap.lapTime):"—";
  label(`LAST  ${lapTime}`,rightX+rightW-12,rightY+40,12,white,700,"right");
  g.strokeStyle=line;g.beginPath();g.moveTo(rightX+10,rightY+56);g.lineTo(rightX+rightW-10,rightY+56);g.stroke();
  const sectors=lap&&lap.sectors?lap.sectors:[null,null,null],cellW=(rightW-24)/3;
  for(let i=0;i<3;i++){
    const x=rightX+12+i*cellW,best=sessionSectorBests[i],isPurple=best&&best.driver.number===driver.number&&lap&&best.lap.lap===lap.lap;
    label(`S${i+1}`,x,rightY+73,9,muted,700);
    label(sectors[i]?formatLapTime(sectors[i]):"—",x,rightY+99,11,isPurple?"#c084fc":white,700);
  }
  const boxW=rightW,boxH=72,boxX=rightX,boxY=h-boxH-14;
  panel(boxX,boxY,boxW,boxH);
  const speed=sample&&sample[1]!=null?Math.round(sample[1]):"—",rpm=sample&&sample[4]!=null?Math.round(sample[4]):"—",gear=sample&&sample[5]!=null?Math.round(sample[5]):"—";
  const aeroYear=data.year>=2026,mode=aeroYear?(sample&&sample[7]===1?"ON":"OFF"):(sample&&sample[6]>0?"ON":"OFF");
  const cell=(x,title,value,color=white)=>{label(title,x,boxY+17,8,muted,700);label(value,x,boxY+43,16,color,800);};
  const gap=(boxW-24)/4;cell(boxX+12,"POS",lap&&lap.pos?String(lap.pos):"—",driver.color||white);cell(boxX+12+gap,"SPEED",`${speed}`,"#ffd166");cell(boxX+12+gap*2,"RPM",`${rpm}`,"#c084fc");cell(boxX+12+gap*3,aeroYear?"GEAR · AERO":"GEAR · DRS",`${gear} / ${mode}`,mode==="ON"?"#48d597":white);
}
function drawPov(t){
  const w=povCanvas.clientWidth,h=povCanvas.clientHeight;
  if(!w||!h)return;
  const dpr=window.devicePixelRatio||1;
  if(povCanvas.width!==Math.round(w*dpr)||povCanvas.height!==Math.round(h*dpr)){
    povCanvas.width=Math.round(w*dpr);povCanvas.height=Math.round(h*dpr);povCtx.setTransform(dpr,0,0,dpr,0,0);
  }
  const g=povCtx;
  g.clearRect(0,0,w,h);
  if(povCameraSelect.value==="overview"){drawTrack3D(t,w,h);return;}
  const sky=g.createLinearGradient(0,0,0,h*.62);sky.addColorStop(0,"#102238");sky.addColorStop(.72,"#71889a");sky.addColorStop(1,"#c6b9a2");g.fillStyle=sky;g.fillRect(0,0,w,h*.62);
  const sun=g.createRadialGradient(w*.78,h*.19,2,w*.78,h*.19,h*.25);sun.addColorStop(0,"rgba(255,244,210,.35)");sun.addColorStop(1,"rgba(255,244,210,0)");g.fillStyle=sun;g.fillRect(0,0,w,h*.55);
  // Layered distant terrain and tree lines give every circuit a real horizon.
  for(let layer=0;layer<3;layer++){
    const base=h*(.405+layer*.027);g.beginPath();g.moveTo(0,h*.55);
    for(let x=0;x<=w+40;x+=40){const y=base+Math.sin(x*.009+layer*2.1)*h*(.018+layer*.006)+Math.sin(x*.021+layer)*h*.009;g.lineTo(x,y);}
    g.lineTo(w,h*.59);g.closePath();g.fillStyle=["#46584f","#34463d","#27372f"][layer];g.fill();
  }
  const horizon=h*.43;
  const ground=g.createLinearGradient(0,horizon,0,h);ground.addColorStop(0,"#53634f");ground.addColorStop(.48,"#38483b");ground.addColorStop(1,"#202a25");g.fillStyle=ground;g.fillRect(0,horizon,w,h-horizon);
  drawPovEnvironment(g,w,h,horizon);
  const driver=data.drivers.find(d=>d.number===selected);
  if(!driver){g.fillStyle="#d7dee8";g.font="600 15px system-ui";g.textAlign="center";g.textBaseline="middle";g.fillText("Choose a driver to generate their 3D viewpoint",w/2,h*.48);g.fillStyle="#929dad";g.font="12px system-ui";g.fillText("The view is reconstructed from the recorded circuit trace",w/2,h*.48+26);return;}
  const car=smoothPositionAt(driver,t)||positionAt(driver,t),located=trackProjection(car);
  if(!located||!trackLength){g.fillStyle="#d7dee8";g.font="14px system-ui";g.textAlign="center";g.fillText("No circuit position is available for this driver at this time",w/2,h*.5);return;}
  const lap=lapAt(driver,t),telem=driver.telemetry||[];let sample=null;
  for(const row of telem){if(row[0]!==null&&row[0]<=t)sample=row;else if(row[0]!==null&&row[0]>t)break;}
  const ahead=smoothPositionAt(driver,t+.22),behind=smoothPositionAt(driver,t-.12);
  const targetHeading=(()=>{const from=behind||car,to=ahead||positionAt(driver,t+.22);if(!from||!to)return [located.tx,located.ty];const dx=to[0]-from[0],dy=to[1]-from[1],length=Math.hypot(dx,dy);return length>0?[dx/length,dy/length]:[located.tx,located.ty];})();
  if(!povHeadingState||povHeadingTime===null||Math.abs(t-povHeadingTime)>1.5){povHeadingState=targetHeading;povHeadingTime=t;}
  else if(t>povHeadingTime){const blend=1-Math.exp(-(t-povHeadingTime)*7),x=povHeadingState[0]*(1-blend)+targetHeading[0]*blend,y=povHeadingState[1]*(1-blend)+targetHeading[1]*blend,len=Math.hypot(x,y)||1;povHeadingState=[x/len,y/len];povHeadingTime=t;}
  const heading=povHeadingState;
  const focal=Math.min(w*.9,h*1.65),cameraHeight=15,halfRoad=62,spacing=Math.max(75,trackLength/480);
  const centerX=w*.5,origin={x:car[0],y:car[1],tx:heading[0],ty:heading[1]};
  function projectWorld(x,y,z=0){
    const dx=x-origin.x,dy=y-origin.y,forward=dx*origin.tx+dy*origin.ty;
    if(forward<24)return null;
    const lateral=dx*origin.ty-dy*origin.tx,scale=focal/(forward+Math.max(0,forward)*.012);
    return {x:centerX+lateral*scale,y:horizon+(cameraHeight-z)*scale,scale,forward};
  }
  // The road mesh is regenerated from this circuit's own recorded centerline.
  const stations=[];
  for(let d=90;d<=Math.min(trackLength,24000);d+=spacing){
    const p=trackPointAt(located.distanceAlong+d);if(!p)continue;
    const deltaX=p.x-origin.x,deltaY=p.y-origin.y,forward=deltaX*origin.tx+deltaY*origin.ty;
    if(forward<20)continue;
    const lateral=deltaX*origin.ty-deltaY*origin.tx,z=forward+Math.max(0,d-forward)*.012,scale=focal/z;
    const center={x:centerX+lateral*scale,y:horizon+cameraHeight*scale,z,scale};
    const tangentRightX=p.ty,tangentRightY=-p.tx;
    const edge=(side,width)=>{
      const ex=p.x+tangentRightX*side*width-origin.x,ey=p.y+tangentRightY*side*width-origin.y;
      const f=ex*origin.tx+ey*origin.ty,lat=ex*origin.ty-ey*origin.tx;
      if(f<20)return null;const sc=focal/(f+Math.max(0,d-f)*.012);
      return {x:centerX+lat*sc,y:horizon+cameraHeight*sc,z:f,scale:sc};
    };
    const left=edge(-1,halfRoad),right=edge(1,halfRoad),curbLeft=edge(-1,halfRoad+13),curbRight=edge(1,halfRoad+13),wallLeft=edge(-1,halfRoad+155),wallRight=edge(1,halfRoad+155);
    if(left&&right)stations.push({center,left,right,curbLeft,curbRight,wallLeft,wallRight,d});
  }
  for(let i=stations.length-1;i>0;i--){
    const far=stations[i],near=stations[i-1];
    g.beginPath();g.moveTo(far.left.x,far.left.y);g.lineTo(far.right.x,far.right.y);g.lineTo(near.right.x,near.right.y);g.lineTo(near.left.x,near.left.y);g.closePath();g.fillStyle=i%4===0?"#41464b":"#383e44";g.fill();
    g.globalAlpha=.2;g.fillStyle=povAsphaltPattern;g.fill();g.globalAlpha=1;
    // Fine asphalt seams and intermittent rubber streaks provide surface scale.
    if(i%7===0){g.beginPath();g.moveTo(far.left.x,far.left.y);g.lineTo(far.right.x,far.right.y);g.strokeStyle="rgba(12,15,18,.34)";g.lineWidth=Math.max(1,near.center.scale*2);g.stroke();}
    for(const side of ["left","right"]){
      const a=far[side],b=near[side],curbA=far[side==="left"?"curbLeft":"curbRight"],curbB=near[side==="left"?"curbLeft":"curbRight"];
      g.beginPath();g.moveTo(a.x,a.y);g.lineTo(b.x,b.y);g.strokeStyle="rgba(241,242,237,.8)";g.lineWidth=Math.max(1,Math.min(3,b.scale*2));g.stroke();
      if(curbA&&curbB){g.beginPath();g.moveTo(curbA.x,curbA.y);g.lineTo(curbB.x,curbB.y);g.strokeStyle=(i%4<2)?"#e7e3d8":"#cf3341";g.lineWidth=Math.max(2,Math.min(14,b.scale*12));g.stroke();}
    }
    for(const side of ["wallLeft","wallRight"]){
      const a=far[side],b=near[side];if(!a||!b)continue;
      const wallHeight=Math.max(2,Math.min(50,b.scale*52));
      g.beginPath();g.moveTo(a.x,a.y);g.lineTo(b.x,b.y);g.lineTo(b.x,b.y-wallHeight);g.lineTo(a.x,a.y-wallHeight);g.closePath();g.fillStyle=i%5===0?"#4e5963":"#343d46";g.fill();
      g.strokeStyle="rgba(205,216,225,.55)";g.lineWidth=Math.max(1,Math.min(2.5,b.scale*2));g.beginPath();g.moveTo(a.x,a.y-wallHeight);g.lineTo(b.x,b.y-wallHeight);g.stroke();
      // Catch fencing above the barrier, represented with perspective-scaled uprights and mesh.
      const fenceHeight=Math.min(92,b.scale*90),fenceTopA=a.y-wallHeight-fenceHeight,fenceTopB=b.y-wallHeight-fenceHeight;
      g.strokeStyle="rgba(183,196,205,.28)";g.lineWidth=Math.max(.6,Math.min(1.5,b.scale));g.beginPath();g.moveTo(a.x,a.y-wallHeight);g.lineTo(a.x,fenceTopA);g.lineTo(b.x,fenceTopB);g.lineTo(b.x,b.y-wallHeight);g.stroke();
      if(i%3===0){g.beginPath();g.moveTo(a.x,fenceTopA+(fenceHeight*.35));g.lineTo(b.x,fenceTopB+(fenceHeight*.35));g.moveTo(a.x,fenceTopA+(fenceHeight*.7));g.lineTo(b.x,fenceTopB+(fenceHeight*.7));g.stroke();}
    }
    if(i%3===0){
      for(const fraction of [.34,.66]){
        const fa=far.left,fb=far.right,na=near.left,nb=near.right;
        const p0=[fa.x+(fb.x-fa.x)*fraction,fa.y+(fb.y-fa.y)*fraction],p1=[na.x+(nb.x-na.x)*fraction,na.y+(nb.y-na.y)*fraction];
        g.beginPath();g.moveTo(...p0);g.lineTo(...p1);g.strokeStyle="rgba(12,14,17,.35)";g.lineWidth=Math.max(.8,Math.min(2.4,near.center.scale*1.6));g.stroke();
      }
    }
  }
  // Distance boards sit at fixed 500 m stations on the circuit. Their labels
  // count down to each station, avoiding the old arbitrary 200/150/100/50 loop.
  const nextStation=(Math.floor(located.distanceAlong/500)+1)*500;
  for(let n=0;n<6;n++){
    const station=nextStation+n*500,stationAlong=station%trackLength;
    let distanceAhead=(stationAlong-located.distanceAlong+trackLength)%trackLength;
    if(distanceAhead<35)distanceAhead+=trackLength;
    if(distanceAhead>2600)continue;
    const p=trackPointAt(stationAlong);if(!p)continue;
    const worldX=p.x+p.ty*105,worldY=p.y-p.tx*105,board=projectWorld(worldX,worldY,0);
    if(!board||board.scale<.025)continue;
    const bw=Math.max(9,Math.min(64,38*board.scale)),bh=Math.max(10,Math.min(44,25*board.scale)),bx=board.x-bw/2,by=board.y-bh;
    g.strokeStyle="#252b31";g.lineWidth=Math.max(1,board.scale*2.4);g.beginPath();g.moveTo(board.x,board.y);g.lineTo(board.x,by+bh*.12);g.stroke();
    g.fillStyle="#e6e8e8";g.fillRect(bx,by,bw,bh);g.strokeStyle="#20252b";g.lineWidth=Math.max(.7,board.scale);g.strokeRect(bx,by,bw,bh);
    g.fillStyle="#18202a";g.textAlign="center";g.textBaseline="middle";g.font=`800 ${Math.max(6,Math.min(15,bh*.55))}px system-ui`;g.fillText(`${Math.round(distanceAhead/100)*100}`,board.x,by+bh*.52);
  }
  // Render rivals between the road and the cockpit, in perspective order. The
  // recorded coordinates determine their positions; distant cars scale down.
  const visibleCars=[];
  for(const rival of data.drivers){
    if(rival.number===driver.number||(rival.dnf&&rival.dnf.time!==null&&t>=rival.dnf.time))continue;
    const rivalPosition=smoothPositionAt(rival,t)||positionAt(rival,t);if(!rivalPosition)continue;
    const dx=rivalPosition[0]-car[0],dy=rivalPosition[1]-car[1];
    const forward=dx*heading[0]+dy*heading[1],lateral=dx*heading[1]-dy*heading[0];
    if(forward<28||forward>1900||Math.abs(lateral)>halfRoad*1.65)continue;
    const scale=focal/(forward+forward*.012),screenX=centerX+lateral*scale,screenY=horizon+cameraHeight*scale;
    if(screenX<-30||screenX>w+30||screenY<0||screenY>h*.79)continue;
    const rivalAhead=smoothPositionAt(rival,t+.18)||rivalPosition,rivalBehind=smoothPositionAt(rival,t-.12)||rivalPosition;
    const rdx=rivalAhead[0]-rivalBehind[0],rdy=rivalAhead[1]-rivalBehind[1];
    const angle=Math.atan2(rdx*heading[1]-rdy*heading[0],rdx*heading[0]+rdy*heading[1]);
    visibleCars.push({rival,screenX,screenY,scale,angle,forward});
  }
  visibleCars.sort((a,b)=>b.forward-a.forward);
  for(const item of visibleCars){
    const {rival,screenX,screenY,scale,angle}=item,width=Math.max(2.2,2.05*scale),length=Math.max(4.5,5.1*scale);
    if(width<2.4)continue;
    g.save();g.translate(screenX,screenY);g.rotate(angle);
    g.fillStyle="rgba(0,0,0,.55)";g.beginPath();g.ellipse(0,length*.12,width*.8,length*.48,0,0,Math.PI*2);g.fill();
    g.fillStyle="#080a0d";
    for(const side of [-1,1])for(const axle of [-1,1]){g.beginPath();g.roundRect(side*width*.53-width*.17,axle*length*.27-length*.105,width*.34,length*.21,Math.max(1,width*.08));g.fill();}
    g.beginPath();g.moveTo(0,-length*.53);g.lineTo(width*.34,-length*.34);g.lineTo(width*.4,length*.25);g.lineTo(width*.28,length*.48);g.lineTo(-width*.28,length*.48);g.lineTo(-width*.4,length*.25);g.lineTo(-width*.34,-length*.34);g.closePath();
    const rivalBody=g.createLinearGradient(-width*.45,0,width*.45,0);rivalBody.addColorStop(0,"#171c23");rivalBody.addColorStop(.22,rival.color||"#8793a3");rivalBody.addColorStop(.5,"#f1f3f4");rivalBody.addColorStop(.76,rival.color||"#8793a3");rivalBody.addColorStop(1,"#171c23");g.fillStyle=rivalBody;g.fill();
    g.strokeStyle="rgba(8,12,18,.9)";g.lineWidth=Math.max(.7,width*.07);g.stroke();
    g.fillStyle="#10151c";g.beginPath();g.ellipse(0,-length*.08,width*.2,length*.15,0,0,Math.PI*2);g.fill();
    g.strokeStyle="#87919a";g.lineWidth=Math.max(.6,width*.055);g.beginPath();g.ellipse(0,-length*.08,width*.32,length*.22,0,Math.PI,Math.PI*2);g.stroke();
    g.fillStyle="#fa334b";g.fillRect(-width*.11,length*.35,width*.22,Math.max(.7,length*.04));g.restore();
    if(scale>1.5){g.font=`700 ${Math.max(7,Math.min(11,scale*2.2))}px system-ui`;g.textAlign="center";g.textBaseline="bottom";g.fillStyle="#f3f5f7";g.shadowColor="#08101a";g.shadowBlur=3;g.fillText(rival.abbr,screenX,screenY-length*.6);g.shadowBlur=0;}
  }
  // Browser-rendered cockpit and car body: the car stays synchronized with
  // replay telemetry and does not depend on a pre-rendered Blender image.
  const teamColor=driver.color||"#8b9bad";
  const rubber=g.createLinearGradient(0,h*.72,0,h);rubber.addColorStop(0,"#30343a");rubber.addColorStop(.38,"#090b0e");rubber.addColorStop(1,"#1d2025");
  for(const x of [w*.145,w*.855]){
    g.save();g.translate(x,h*.865);g.rotate(x<w/2?-.1:.1);
    const sidewall=g.createLinearGradient(-w*.07,0,w*.07,0);sidewall.addColorStop(0,"#050607");sidewall.addColorStop(.32,"#282b2e");sidewall.addColorStop(.58,"#101214");sidewall.addColorStop(1,"#050607");
    g.fillStyle=sidewall;g.beginPath();g.ellipse(0,0,w*.071,h*.14,0,0,Math.PI*2);g.fill();
    g.strokeStyle="rgba(200,205,207,.28)";g.lineWidth=Math.max(2,w*.004);g.beginPath();g.ellipse(0,0,w*.063,h*.13,0,Math.PI*1.05,Math.PI*1.95);g.stroke();
    g.strokeStyle="rgba(220,226,230,.22)";g.lineWidth=1;for(let k=-2;k<=2;k++){g.beginPath();g.moveTo(k*w*.014,-h*.09);g.lineTo(k*w*.014,h*.09);g.stroke();}
    g.restore();
  }
  // Front wing and endplates.
  g.fillStyle="#090d12";g.beginPath();g.moveTo(w*.12,h*.925);g.lineTo(w*.29,h*.91);g.lineTo(w*.43,h*.92);g.lineTo(w*.5,h*.935);g.lineTo(w*.57,h*.92);g.lineTo(w*.71,h*.91);g.lineTo(w*.88,h*.925);g.lineTo(w*.84,h*.956);g.lineTo(w*.16,h*.956);g.closePath();g.fill();
  g.fillStyle=teamColor;g.beginPath();g.moveTo(w*.16,h*.944);g.lineTo(w*.35,h*.93);g.lineTo(w*.5,h*.947);g.lineTo(w*.65,h*.93);g.lineTo(w*.84,h*.944);g.lineTo(w*.82,h*.956);g.lineTo(w*.18,h*.956);g.closePath();g.fill();
  g.fillStyle="#171e27";g.fillRect(w*.115,h*.91,w*.035,h*.055);g.fillRect(w*.85,h*.91,w*.035,h*.055);
  // Carbon bodywork and tapered nose.
  const carbon=g.createLinearGradient(0,h*.74,0,h);carbon.addColorStop(0,"#262d35");carbon.addColorStop(.22,"#080b10");carbon.addColorStop(.72,"#151a20");carbon.addColorStop(1,"#05070a");
  g.fillStyle=carbon;g.beginPath();g.moveTo(0,h);g.lineTo(0,h*.92);g.quadraticCurveTo(w*.5,h*.76,w,h*.92);g.lineTo(w,h);g.closePath();g.fill();
  const body=g.createLinearGradient(w*.42,h*.83,w*.59,h);body.addColorStop(0,"#080b10");body.addColorStop(.38,"#252c34");body.addColorStop(.52,"#10151c");body.addColorStop(.72,"#252c34");body.addColorStop(1,"#080b10");
  g.fillStyle=body;g.beginPath();g.moveTo(w*.34,h);g.lineTo(w*.44,h*.84);g.quadraticCurveTo(w*.5,h*.80,w*.56,h*.84);g.lineTo(w*.66,h);g.closePath();g.fill();
  const livery=g.createLinearGradient(0,h*.83,0,h);livery.addColorStop(0,teamColor);livery.addColorStop(.45,teamColor);livery.addColorStop(1,"rgba(7,10,14,.95)");
  g.fillStyle=livery;g.beginPath();g.moveTo(w*.43,h);g.lineTo(w*.48,h*.86);g.lineTo(w*.5,h*.83);g.lineTo(w*.52,h*.86);g.lineTo(w*.57,h);g.closePath();g.fill();
  g.strokeStyle="rgba(255,255,255,.42)";g.lineWidth=Math.max(1,w*.0018);g.beginPath();g.moveTo(w*.485,h*.86);g.lineTo(w*.5,h*.835);g.lineTo(w*.515,h*.86);g.stroke();
  // Suspension arms, mirrors, and a dark titanium halo.
  g.strokeStyle="#777f87";g.lineWidth=Math.max(1.4,w*.0025);for(const side of [-1,1]){g.beginPath();g.moveTo(w*.5,h*.835);g.lineTo(w*(.5+side*.24),h*.78);g.moveTo(w*(.5+side*.08),h*.86);g.lineTo(w*(.5+side*.31),h*.82);g.stroke();}
  g.fillStyle="#101720";for(const side of [-1,1]){g.beginPath();g.ellipse(w*(.5+side*.285),h*.765,w*.025,h*.012,side*.15,0,Math.PI*2);g.fill();g.strokeStyle="#77828b";g.lineWidth=1;g.stroke();}
  const halo=g.createLinearGradient(0,h*.58,0,h*.81);halo.addColorStop(0,"#aeb7bc");halo.addColorStop(.18,"#252c33");halo.addColorStop(.6,"#090c10");halo.addColorStop(1,"#69737b");
  g.strokeStyle="#05070a";g.lineWidth=Math.max(11,w*.017);g.lineCap="round";g.beginPath();g.moveTo(w*.245,h*.79);g.quadraticCurveTo(w*.5,h*.50,w*.755,h*.79);g.stroke();
  g.strokeStyle=halo;g.lineWidth=Math.max(7,w*.010);g.beginPath();g.moveTo(w*.245,h*.79);g.quadraticCurveTo(w*.5,h*.50,w*.755,h*.79);g.stroke();
  g.strokeStyle="#151b21";g.lineWidth=Math.max(6,w*.008);g.beginPath();g.moveTo(w*.5,h*.56);g.lineTo(w*.5,h*.845);g.stroke();
  g.strokeStyle="#737d84";g.lineWidth=Math.max(2,w*.002);g.beginPath();g.moveTo(w*.5,h*.57);g.lineTo(w*.5,h*.82);g.stroke();
  // Steering wheel, telemetry display, and rev lights.
  g.strokeStyle="#05070a";g.lineWidth=Math.max(8,w*.012);g.beginPath();g.moveTo(w*.435,h*.89);g.quadraticCurveTo(w*.5,h*.86,w*.565,h*.89);g.lineTo(w*.55,h*.955);g.quadraticCurveTo(w*.5,h*.975,w*.45,h*.955);g.closePath();g.stroke();
  g.fillStyle="#0a1017";g.beginPath();g.roundRect(w*.455,h*.895,w*.09,h*.06,5);g.fill();g.strokeStyle=teamColor;g.lineWidth=1.5;g.stroke();
  const gear=sample&&sample[5]!=null?Math.round(sample[5]):null,speed=sample&&sample[1]!=null?Math.round(sample[1]):null;
  g.textAlign="center";g.textBaseline="middle";g.fillStyle="#fff";g.font="800 18px ui-monospace,monospace";g.fillText(gear===null?"N":String(gear),w*.5,h*.916);
  g.font="8px system-ui";g.fillStyle="#b9c4d1";g.fillText(speed===null?"— km/h":`${speed} km/h`,w*.5,h*.941);
  const rpm=sample&&sample[4]!=null?Number(sample[4]):0,leds=Math.max(0,Math.min(10,Math.round(rpm/15000*10)));
  for(let i=0;i<10;i++){g.fillStyle=i<leds?(i<6?"#38d996":i<8?"#ffc857":"#ff4c5d"):"#26313d";g.fillRect(w*.455+i*w*.009,h*.88,w*.006,h*.007);}
  for(let side of [-1,1]){const bx=w*(.5+side*.049),by=h*.921;for(let k=0;k<3;k++){g.beginPath();g.arc(bx,by+(k-1)*h*.011,Math.max(1.5,w*.0025),0,Math.PI*2);g.fillStyle=["#ef5965","#ffd166","#49a6ff"][k];g.fill();}}
  const cardW=Math.min(w-32,360);g.fillStyle="rgba(8,13,20,.9)";g.beginPath();g.roundRect(16,16,cardW,74,12);g.fill();g.strokeStyle="rgba(178,192,209,.3)";g.stroke();
  drawTeamMark(g,28,31,driver.team,teamColor,driver.logoKey);
  g.textAlign="left";g.textBaseline="top";g.fillStyle="#f5f7fa";g.font="800 16px system-ui";g.fillText(driver.name,88,30,cardW-98);
  g.fillStyle=teamColor;g.font="600 12px system-ui";g.fillText(driver.team||"Team",88,53,cardW-98);
  g.fillStyle="#aab6c5";g.font="11px system-ui";const lapLabel=lap&&lap.lap?`LAP ${lap.lap}`:"LAP —";g.fillText(`${lapLabel}  ·  ${speed===null?"— km/h":speed+" km/h"}`,88,70,cardW-98);
  drawPovHud(t,driver,lap,sample,w,h);
}
function syncDriverOptions(select,orderedDrivers){
  // Keep an open native dropdown stable while the replay advances.
  if(document.activeElement===select)return;
  const ordered=orderedDrivers.length?orderedDrivers:data.drivers;
  const numbers=ordered.map(driver=>String(driver.number));
  const current=[...select.options].slice(1).map(option=>option.value);
  if(numbers.length===current.length&&numbers.every((number,index)=>number===current[index]))return;
  const selectedValue=select.value;
  select.innerHTML='<option value="">Choose a driver</option>'+ordered.map(driver=>
    `<option value="${driver.number}">${escapeHtml(driver.abbr)} · ${escapeHtml(driver.name)}</option>`
  ).join("");
  select.value=selectedValue;
}
function syncTelemetryDriverOptions(orderedDrivers){
  syncDriverOptions(telemetryDriverSelect,orderedDrivers);
  syncDriverOptions(povDriverSelect,orderedDrivers);
}
function updateRacePanels(t){
  if(!isRace){
    const ranked=data.drivers.map(d=>{
      const completed=d.laps.filter(l=>l.t!==null&&l.t<=t&&l.lapTime!==null&&l.lapTime>0&&l.valid!==false);
      const best=completed.length?Math.min(...completed.map(l=>l.lapTime)):null;
      return {d,best,lap:lapAt(d,t)};
    }).sort((a,b)=>{
      if(a.best===null&&b.best!==null)return 1;
      if(a.best!==null&&b.best===null)return -1;
      if(a.best!==null&&b.best!==null&&a.best!==b.best)return a.best-b.best;
      return a.d.number-b.d.number;
    });
    const sessionBest=ranked.find(row=>row.best!==null)?.best??null;
    timingTower.innerHTML=ranked.map((row,index)=>{
      const best=formatLapTime(row.best);
      const gap=row.best===null||sessionBest===null?"—":(row.best-sessionBest).toFixed(3)+"s";
      const improved=row.lap&&isPersonalBest(row.d,row.lap);
      const lastLap=row.lap&&row.lap.lapTime!==null?formatLapTime(row.lap.lapTime)+(improved?" · PB":""):"—";
      const tyre=row.lap&&row.lap.tyre?escapeHtml(row.lap.tyre):"—";
      return `<div class="tower-row ${row.d.number===selected?"selected-driver":""}" data-driver-number="${row.d.number}"><span class="pos">${index+1}</span><span class="abbr" style="color:${escapeHtml(row.d.color)}">${escapeHtml(row.d.abbr)}</span><span class="lap-info">${best}</span><span class="gap">${gap}</span><span class="lap-info ${improved?"green":""}">${lastLap}</span><span class="delta">${lapDeltaAt(row.d,t)}</span><span class="tyre">${tyre}</span></div>`;
    }).join("")||'<div class="empty">No timing data available</div>';
    syncTelemetryDriverOptions(ranked.map(row=>row.d));
    lapCount.textContent="";
    return;
  }
  const classifiedOut=data.drivers.filter(d=>d.dnf&&d.dnf.time!==null&&t>=d.dnf.time);
  const running=data.drivers.filter(d=>!(d.dnf&&d.dnf.time!==null&&t>=d.dnf.time)).map(d=>{
    const lap=lapAt(d,t),position=positionAt(d,t),lapNumber=lap&&lap.lap!==null?lap.lap:0;
    return {d,lap,progress:lapNumber+trackProgress(position)};
  });
  running.sort((a,b)=>b.progress-a.progress);
  let completedLaps=0;
  for(const d of data.drivers)for(const lap of d.laps)if(lap.t!==null&&lap.t<=t&&lap.lap!==null)completedLaps=Math.max(completedLaps,lap.lap);
  if(data.session==="Race"&&data.totalLaps){
    const currentRaceLap=completedLaps===0?1:Math.min(data.totalLaps,completedLaps+1);
    lapCount.textContent=`Lap ${currentRaceLap}/${data.totalLaps}`;
  }else lapCount.textContent="";
  const lapPaces=running.map(row=>row.lap&&row.lap.lapTime).filter(x=>x!==null&&x>0);
  const typicalLap=lapPaces.length?lapPaces.reduce((sum,x)=>sum+x,0)/lapPaces.length:90;
  const activeRows=running.map((item,index)=>{
    const ahead=running[index-1],gap=index===0?"0.0":ahead?Math.max(0,(ahead.progress-item.progress)*typicalLap).toFixed(1)+"s":"—";
    const improved=item.lap&&isPersonalBest(item.d,item.lap);
    const lapTime=item.lap&&item.lap.lapTime!==null?formatLapTime(item.lap.lapTime)+(improved?" · PB":""):"—";
    const tyre=item.lap&&item.lap.tyre?escapeHtml(item.lap.tyre):"—";
    return `<div class="tower-row ${item.d.number===selected?"selected-driver":""}" data-driver-number="${item.d.number}"><span class="pos">${index+1}</span><span class="abbr" style="color:${escapeHtml(item.d.color)}">${escapeHtml(item.d.abbr)}</span><span class="current-lap">${lapNumberAt(item.d,t)}</span><span class="gap">${gap}</span><span class="lap-info ${improved?"green":""}">${lapTime}</span><span class="delta">${lapDeltaAt(item.d,t)}</span><span class="tyre">${tyre}</span></div>`;
  });
  const dnfRows=classifiedOut.map((d,index)=>
    `<div class="tower-row dnf-entry ${d.number===selected?"selected-driver":""}" data-driver-number="${d.number}"><span class="pos">${running.length+index+1}</span><span class="abbr" style="color:${escapeHtml(d.color)}">${escapeHtml(d.abbr)}</span><span class="current-lap">${lapNumberAt(d,t)}</span><span class="gap">-</span><span class="lap-info">-</span><span class="delta">-</span><span class="tyre">-</span></div>`
  );
  syncTelemetryDriverOptions(running.map(row=>row.d).concat(classifiedOut));
  timingTower.innerHTML=activeRows.concat(dnfRows).join("")||'<div class="empty">No timing data available</div>';
}
function isPersonalBest(driver,lap){
  if(!lap||lap.lapTime===null||lap.t===null||lap.valid===false)return false;
  const previous=driver.laps.filter(item=>item.t!==null&&item.t<lap.t&&item.lapTime!==null&&item.lapTime>0&&item.valid!==false);
  return !previous.length||lap.lapTime<Math.min(...previous.map(item=>item.lapTime))-0.0005;
}
function formatLapTime(seconds){
  if(seconds===null||seconds===undefined||!Number.isFinite(Number(seconds)))return "—";
  const totalMs=Math.round(Number(seconds)*1000);
  const minutes=Math.floor(totalMs/60000);
  const wholeSeconds=Math.floor((totalMs%60000)/1000);
  const milliseconds=totalMs%1000;
  return `${minutes}:${String(wholeSeconds).padStart(2,"0")}:${String(milliseconds).padStart(3,"0")}`;
}
function updateSessionSectorBests(t){
  const next=[0,1,2].map(index=>{
    let result=null;
    for(const driver of data.drivers){
      for(const lap of driver.laps){
        if(lap.valid===false)continue;
        const sectors=lap.sectors||[],time=sectors[index];
        if(time===null||time===undefined||time<=0||sectors.slice(0,index+1).some(value=>value===null||value===undefined||value<=0))continue;
        const lapStart=lap.lapStart!==null&&lap.lapStart!==undefined?lap.lapStart:
          (lap.t!==null&&lap.t!==undefined&&lap.lapTime!==null&&lap.lapTime!==undefined?lap.t-lap.lapTime:null);
        if(lapStart===null)continue;
        const sectorEnd=lapStart+sectors.slice(0,index+1).reduce((sum,value)=>sum+value,0);
        if(sectorEnd>t)continue;
        if(!result||time<result.time)result={driver,time,lap};
      }
    }
    return result;
  });
  const changed=next.some((best,index)=>{
    const previous=sessionSectorBests[index];
    return (best?.driver.number??null)!==(previous?.driver.number??null)||
      (best?.lap.lap??null)!==(previous?.lap.lap??null)||
      Math.abs((best?.time??0)-(previous?.time??0))>0.0005;
  });
  if(changed)sessionSectorBests=next;
  return changed;
}
function updateSessionHighlights(t){
  const validLaps=[];
  for(const driver of data.drivers){
    let best=Infinity;
    for(const lap of driver.laps){
      if(lap.t===null||lap.t>t||lap.valid===false||lap.lapTime===null||lap.lapTime<=0)continue;
      validLaps.push({driver,lap});
      best=Math.min(best,lap.lapTime);
    }
  }
  const fastest=validLaps.slice().sort((a,b)=>a.lap.lapTime-b.lap.lapTime)[0];
  const recentImprovements=[];
  for(const driver of data.drivers){
    let previousBest=Infinity;
    for(const lap of driver.laps){
      if(lap.t===null||lap.t>t||lap.valid===false||lap.lapTime===null||lap.lapTime<=0)continue;
      if(lap.lapTime<previousBest-0.0005&&lap.t>=t-60)recentImprovements.push({driver,lap});
      previousBest=Math.min(previousBest,lap.lapTime);
    }
  }
  recentImprovements.sort((a,b)=>b.lap.t-a.lap.t);
  let spotlight=null;
  if(isRace){
    for(const driver of data.drivers){
      if(driver.dnf&&driver.dnf.time!==null&&t>=driver.dnf.time)continue;
      const lap=lapAt(driver,t),position=lap&&lap.pos;
      if(driver.grid===null||position===null||position===undefined)continue;
      const gain=driver.grid-position;
      if(!spotlight||gain>spotlight.gain)spotlight={driver,gain};
    }
  }
  const fmt=formatLapTime;
  const card=(label,value,sub="",cls="")=>`<div class="highlight-card"><div class="highlight-label">${label}</div><div class="highlight-value ${cls}">${value}</div><div class="highlight-sub">${sub}</div></div>`;
  const sectorCards=sessionSectorBests.map((best,index)=>card(
    `Purple · S${index+1}`,best?best.driver.abbr:"—",best?fmt(best.time):"No sector yet","purple"
  ));
  const improvement=recentImprovements[0];
  const improvementCard=card("Lap improvements · 60s",improvement?improvement.driver.abbr:"—",improvement?`Lap ${improvement.lap.lap} · ${fmt(improvement.lap.lapTime)}`:"No new personal bests");
  const raceFinished=isRace&&t>=data.start+data.duration-1;
  let spotlightCard;
  if(raceFinished){
    spotlightCard=data.officialDotd
      ?card("Official Driver of the Day",escapeHtml(data.officialDotd.name),`Fan-voted award · ${escapeHtml(data.officialDotd.abbr)}`)
      :card("Official Driver of the Day","Unavailable","Award result not found in the data source");
  }else{
    spotlightCard=card(isRace?"Driver of the day · estimate":"Session leader",isRace?(spotlight?spotlight.driver.abbr:"—"):(fastest?fastest.driver.abbr:"—"),isRace?(spotlight?`${spotlight.gain>0?"+":""}${spotlight.gain} places gained`:"No classified positions yet"):"Fastest lap holder");
  }
  document.getElementById("session-highlights").innerHTML=[
    card("Fastest lap",fastest?fastest.driver.abbr:"—",fastest?fmt(fastest.lap.lapTime):"No valid lap yet"),
    ...sectorCards,
    improvementCard,
    spotlightCard,
  ].join("");
}
function renderDriverDetails(t,force=false){
  const driver=data.drivers.find(d=>d.number===selected);
  if(!driver){detail.classList.remove("open");lastPhotoDriver=null;return;}
  const second=Math.floor(t);
  if(!force&&second===lastDetailSecond)return;
  lastDetailSecond=second;detail.classList.add("open");
  driverAbbr.textContent=driver.abbr;driverAbbr.style.color=driver.color;
  driverName.textContent=driver.name;driverTeam.textContent=driver.team;
  const out=driver.dnf&&driver.dnf.time!==null&&t>=driver.dnf.time;
  const dns=out&&/did not start|\bDNS\b/i.test(driver.dnf.status||"");
  driverState.textContent=dns?"DNS":out?"DNF · Lap "+(driver.dnf.lap??"—"):"On track";
  if(lastPhotoDriver!==driver.number){
    lastPhotoDriver=driver.number;
    if(driver.photo){driverPhoto.hidden=false;photoFallback.hidden=true;driverPhoto.onerror=()=>{driverPhoto.hidden=true;photoFallback.hidden=false;};driverPhoto.src=driver.photo;}
    else{driverPhoto.hidden=true;photoFallback.hidden=false;}
    photoFallback.textContent=driver.abbr;
  }
  const infoTime=out?driver.dnf.time:t,lap=lapAt(driver,infoTime),lapNum=lapNumberAt(driver,t);
  const position=!out&&lap&&lap.pos!==null?lap.pos:"—";
  const places=!out&&lap&&lap.pos!==null&&driver.grid!==null?((driver.grid-lap.pos>0?"+":"")+(driver.grid-lap.pos)):"—";
  const timedLaps=driver.laps.filter(x=>x.t!==null&&x.t<=infoTime&&x.lapTime!==null&&x.lapTime>0&&x.valid!==false);
  const best=timedLaps.length?Math.min(...timedLaps.map(x=>x.lapTime)):null;
  const metric=(label,value,className="")=>`<div class="metric"><div class="metric-label">${escapeHtml(label)}</div><div class="metric-value ${className}">${escapeHtml(value??"—")}</div></div>`;
  const lapTime=lap&&lap.lapTime!==null?formatLapTime(lap.lapTime):"—";
  const tyre=lap&&lap.tyre?lap.tyre:"—",age=lap&&lap.tyreAge!==null?String(lap.tyreAge)+" laps":"—";
  const driverSectorBests=[0,1,2].map(index=>{
    const values=timedLaps.map(item=>item.sectors?.[index]).filter(value=>value!==null&&value!==undefined&&value>0);
    return values.length?Math.min(...values):null;
  });
  const sectorMetrics=driverSectorBests.map((value,index)=>{
    const purple=sessionSectorBests[index]?.driver.number===driver.number;
    return metric(`Best sector ${index+1}`,formatLapTime(value),purple?"purple":"");
  });
  driverStats.innerHTML=metric("Lap",lapNum)+metric("Position",position)+metric("Grid",driver.grid??"—")+
    metric("Places gained",places)+
    metric("Last lap",out?"—":lapTime)+metric("Tyre",out?"—":tyre)+metric("Tyre age",out?"—":age)+
    metric("Best lap",formatLapTime(best))+
    sectorMetrics.join("");
}
function renderTelemetry(t,force=false){
  const driver=data.drivers.find(d=>d.number===selected);
  const second=Math.floor(t);
  if(!force&&second===lastTelemetrySecond)return;
  lastTelemetrySecond=second;
  if(!driver){telemetryDriver.textContent="Car telemetry";telemetryDriverSelect.value="";telemetryTime.textContent="Select a driver to inspect recorded telemetry";telemetryMetrics.innerHTML="";telemetryEmpty.textContent="Click a driver on the map or timing tower, or choose one above, to see their telemetry.";telemetryEmpty.hidden=false;telemetryStack.hidden=true;return;}
  if(telemetryDriverSelect.value!==String(driver.number))telemetryDriverSelect.value=String(driver.number);
  if(document.activeElement!==driverSearch&&driverSearch.value!==driver.name)driverSearch.value=driver.name;
  telemetryDriver.textContent=driver.abbr+" · "+driver.name+" — "+driver.team;
  const samples=driver.telemetry||[];
  let sample=null;
  for(const row of samples){if(row[0]!==null&&row[0]<=t)sample=row;else if(row[0]!==null&&row[0]>t)break;}
  const lap=lapAt(driver,t),tyreAge=lap&&lap.tyreAge!==null&&lap.tyreAge!==undefined?`${lap.tyreAge} laps`:"—";
  const pretty=(value,suffix="")=>value===null||value===undefined?"—":`${Math.round(value)}${suffix}`;
  const metric=(label,value)=>`<div class="metric"><div class="metric-label">${label}</div><div class="metric-value">${value}</div></div>`;
  telemetryMetrics.innerHTML=metric("Speed",sample?pretty(sample[1]," km/h"):"—")+
    metric("Throttle",sample?pretty(sample[2],"%"):"—")+
    metric("Brake",sample&&sample[3]!==null?(sample[3]?"Applied":"Off"):"—")+
    metric("Gear",sample?pretty(sample[5]):"—")+
    metric("RPM",sample?pretty(sample[4]):"—")+
    metric("Tyre age",tyreAge)+
    metric(data.year>=2026?"Active aero":"DRS",sample?pretty(sample[data.year>=2026?7:6]):"—");
  if(!samples.length){telemetryTime.textContent="Car telemetry unavailable for this session";telemetryEmpty.textContent="FastF1 has no recorded car telemetry samples for this driver in this session.";telemetryEmpty.hidden=false;telemetryStack.hidden=true;return;}
  telemetryTime.textContent="Replay time · "+Math.floor(t/60)+":"+String(Math.floor(t%60)).padStart(2,"0");
  telemetryEmpty.hidden=true;telemetryStack.hidden=false;
  const start=Math.max(data.start,t-60),visible=samples.filter(row=>row[0]!==null&&row[0]>=start&&row[0]<=t),dpr=window.devicePixelRatio||1;
  const activeAeroUnavailable=document.getElementById("active-aero-unavailable");
  const hasActiveAero=samples.some(row=>row[7]!==null&&row[7]!==undefined);
  activeAeroUnavailable.hidden=!(data.year>=2026&&!hasActiveAero);
  for(const chart of telemetryCharts){
    const {canvas,ctx,index,label,color,unit,max}=chart,width=canvas.clientWidth,height=canvas.clientHeight;
    const isDrs=index===6,isActiveAero=index===7;
    const chartIsAvailable=data.year>=2026?(!isDrs&&(!isActiveAero||hasActiveAero)):!isActiveAero;
    canvas.hidden=!chartIsAvailable;
    if(!chartIsAvailable)continue;
    if(!width||!height)continue;
    if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);}
    let chartMax=index===6?1:max;
    const pad={left:62,right:18,top:30,bottom:24},plotW=width-pad.left-pad.right,plotH=height-pad.top-pad.bottom;
    ctx.clearRect(0,0,width,height);
    if(index===6){ctx.fillStyle="rgba(46, 204, 113, .10)";ctx.fillRect(pad.left,pad.top,plotW,plotH/2);ctx.fillStyle="rgba(239, 89, 101, .08)";ctx.fillRect(pad.left,pad.top+plotH/2,plotW,plotH/2);}
    ctx.font="11px system-ui";ctx.textBaseline="middle";ctx.textAlign="left";ctx.fillStyle=color;
    const drsWindow=visible.map(row=>Number(row[6])).filter(Number.isFinite);
    const drsOn=value=>[10,12,14].includes(Math.round(Number(value)));
    const chartTitle=index===6?label+(drsWindow.length&&!drsWindow.some(drsOn)?" · OFF in this 60s window":" · ON / OFF"):label+(unit?" · "+unit:"");
    ctx.fillText(chartTitle,pad.left,15);
    for(let tick=0;tick<=2;tick++){const value=chartMax*(1-tick/2),y=pad.top+plotH*tick/2;ctx.strokeStyle="#26313f";ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(width-pad.right,y);ctx.stroke();ctx.textAlign="right";ctx.fillStyle="#8793a3";const axisLabel=index===3?(tick===0?"On":tick===2?"Off":""):index===6?(tick===0?"ON":tick===2?"OFF":""):Math.round(value).toLocaleString();ctx.fillText(axisLabel,pad.left-8,y);}
    const timeX=sec=>pad.left+(sec-start)/(t-start||1)*plotW;
    if(index===6){
      const drsPoints=visible.filter(row=>row[6]!==null&&row[6]!==undefined).map(row=>({x:timeX(row[0]),on:drsOn(row[6])}));
      for(let i=1;i<drsPoints.length;i++){
        const a=drsPoints[i-1],next=drsPoints[i],ya=pad.top+(a.on?plotH*.12:plotH*.88),yb=pad.top+(next.on?plotH*.12:plotH*.88);
        ctx.beginPath();ctx.moveTo(a.x,ya);ctx.lineTo(next.x,ya);ctx.strokeStyle=a.on?"#35e28a":"#ff5964";ctx.lineWidth=5;ctx.lineCap="butt";ctx.stroke();
        if(a.on!==next.on){ctx.beginPath();ctx.moveTo(next.x,ya);ctx.lineTo(next.x,yb);ctx.strokeStyle=next.on?"#35e28a":"#ff5964";ctx.lineWidth=4;ctx.stroke();}
      }
    }else{
      ctx.beginPath();let begun=false,previous=null;
      for(const row of visible){const value=row[index];if(value===null||value===undefined)continue;const x=timeX(row[0]),y=pad.top+plotH-(Math.max(0,Math.min(chartMax,value))/chartMax)*plotH;if(!begun){ctx.moveTo(x,y);begun=true;}else if(index===5){ctx.lineTo(x,previous[1]);ctx.lineTo(x,y);}else ctx.lineTo(x,y);previous=[x,y];}
      ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();
    }
    for(let tick=0;tick<=4;tick++){const x=pad.left+plotW*tick/4;ctx.strokeStyle="#293442";ctx.beginPath();ctx.moveTo(x,pad.top+plotH);ctx.lineTo(x,pad.top+plotH+3);ctx.stroke();ctx.textAlign="center";ctx.fillStyle="#788596";const ago=Math.round((t-start)*(1-tick/4));ctx.fillText(ago===0?"now":"−"+ago+"s",x,height-9);}
    const marker=sample&&sample[0]!==null?timeX(sample[0]):null;
    if(marker!==null){ctx.setLineDash([4,4]);ctx.strokeStyle="#d7dee8";ctx.beginPath();ctx.moveTo(marker,pad.top);ctx.lineTo(marker,pad.top+plotH);ctx.stroke();ctx.setLineDash([]);}
  }
}
function draw(){
  const w=canvas.clientWidth,h=canvas.clientHeight;
  const t=data.start+current;
  if(updateSessionSectorBests(t)){updateSessionHighlights(t);renderDriverDetails(t,true);}
  const flag=trackStyleAt(t);
  flagState.textContent="Track status · "+flag.label;
  flagState.style.color=flag.color;
  if(w&&h){
  ctx.clearRect(0,0,w,h);const b=boundsCache||bounds();boundsCache=b;
  if(!b){
    ctx.fillStyle="#929dad";ctx.font="14px system-ui";ctx.textAlign="center";ctx.textBaseline="middle";
    ctx.fillText("A reliable circuit outline is unavailable for this session.",w/2,h/2);
  }else{
  ctx.beginPath();data.track.forEach((p,i)=>{const q=xy(p[0],p[1],b);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});
  ctx.strokeStyle=flag.color;ctx.lineWidth=flag.label==="Track clear"?4:6;ctx.lineCap="round";ctx.lineJoin="round";
  if(flag.label!=="Track clear"){ctx.shadowColor=flag.color;ctx.shadowBlur=18;}ctx.stroke();ctx.shadowBlur=0;
  drawSelectedPurpleSectors(b);
  const drawDrivers=[...data.drivers].sort((a,b)=>Number(a.number===selected)-Number(b.number===selected));
  for(const d of drawDrivers){
    if(d.dnf&&d.dnf.time!==null&&t>=d.dnf.time)continue;
    const p=positionAt(d,t);if(!p)continue;
    const q=xy(p[0],p[1],b),isSelected=d.number===selected;
    ctx.beginPath();ctx.arc(q[0],q[1],isSelected?11:7,0,Math.PI*2);
    ctx.fillStyle=d.color;ctx.fill();ctx.lineWidth=isSelected?3:1.5;ctx.strokeStyle=isSelected?"#ff4655":"#f7f9fc";ctx.stroke();
    ctx.font=isSelected?"bold 12px system-ui":"bold 11px system-ui";ctx.textAlign="center";ctx.textBaseline="bottom";
    ctx.fillStyle="#f2f4f8";ctx.shadowColor="#0b1018";ctx.shadowBlur=4;ctx.fillText(d.abbr,q[0],q[1]-(isSelected?30:9));ctx.shadowBlur=0;
  }
  }
  }
  const mins=Math.floor(current/60),secs=(current%60).toFixed(1).padStart(4,"0");
  clock.textContent=mins+":"+secs+" / "+Math.floor(data.duration/60)+":"+String(Math.floor(data.duration%60)).padStart(2,"0");
  const towerSecond=Math.floor(t);
  if(towerSecond!==lastTowerSecond){updateRacePanels(t);updateSessionHighlights(t);lastTowerSecond=towerSecond;}
  renderDriverDetails(t);
  if(viewMode.value==="pov"){
    const now=performance.now();
    if(now-lastPovRenderMs>=40){lastPovRenderMs=now;drawPov(t);}
  }
}
function selectDriver(number){
  selected=number===null?null:Number(number);
  povHeadingState=null;povHeadingTime=null;
  povDriverSelect.value=selected===null?"":String(selected);
  lastDetailSecond=-1;lastTowerSecond=-1;lastTelemetrySecond=-1;
  if(selected===null)driverSearch.value="";
  updateSessionHighlights(data.start+current);
  draw();drawPov(data.start+current);renderDriverDetails(data.start+current,true);renderTelemetry(data.start+current);
}
function loop(timestamp){
  if(playing){
    if(last)current=Math.min(data.duration,current+(timestamp-last)/1000*Number(speedSelect.value));
    last=timestamp;seek.value=current;
    if(current>=data.duration){playing=false;playButton.textContent="▶ Play";}
  }else last=0;
  draw();renderTelemetry(data.start+current);requestAnimationFrame(loop);
}
playButton.onclick=()=>{playing=!playing;playButton.textContent=playing?"❚❚ Pause":"▶ Play";last=0;};
jumpStartButton.onclick=()=>{current=0;playing=false;last=0;povHeadingState=null;povHeadingTime=null;seek.value=0;playButton.textContent="▶ Play";lastTowerSecond=-1;lastDetailSecond=-1;lastTelemetrySecond=-1;draw();renderTelemetry(data.start,true);};
seek.oninput=()=>{current=Number(seek.value);playing=false;last=0;povHeadingState=null;povHeadingTime=null;playButton.textContent="▶ Play";draw();renderTelemetry(data.start+current,true);};
speedSelect.onchange=()=>{last=0;};
canvas.onclick=(ev)=>{
  const rect=canvas.getBoundingClientRect(),mx=ev.clientX-rect.left,my=ev.clientY-rect.top,b=bounds();
  if(!b)return;
  let nearest=null,distance=22;
  for(const d of data.drivers){if(d.dnf&&d.dnf.time!==null&&data.start+current>=d.dnf.time)continue;const p=positionAt(d,data.start+current);if(!p)continue;const q=xy(p[0],p[1],b),dist=Math.hypot(q[0]-mx,q[1]-my);if(dist<distance){distance=dist;nearest=d;}}
  if(nearest)selectDriver(nearest.number);
};
timingTower.onclick=(ev)=>{
  const row=ev.target.closest("[data-driver-number]");
  if(row)selectDriver(Number(row.dataset.driverNumber));
};
document.getElementById("close-detail").onclick=()=>selectDriver(null);
telemetryDriverSelect.onchange=()=>{
  selectDriver(telemetryDriverSelect.value?Number(telemetryDriverSelect.value):null);
};
telemetryDriverSelect.onblur=()=>updateRacePanels(data.start+current);
povDriverSelect.onchange=()=>selectDriver(povDriverSelect.value?Number(povDriverSelect.value):null);
povCameraSelect.onchange=()=>drawPov(data.start+current);
driverSearch.onkeydown=ev=>{if(ev.key==="Enter"){const query=driverSearch.value.trim().toLocaleLowerCase();const matches=data.drivers.filter(d=>d.name.toLocaleLowerCase().includes(query)||d.abbr.toLocaleLowerCase().includes(query));const match=matches.find(d=>d.name.toLocaleLowerCase()===query||d.abbr.toLocaleLowerCase()===query)||(matches.length===1?matches[0]:null);if(match){selectDriver(match.number);ev.preventDefault();}}};
window.addEventListener("resize",resize);resize();requestAnimationFrame(loop);
</script></body></html>""".replace("__DATA__", data_json)

replay_tab, championship_tab = st.tabs(["Replay", "Championship"])
with replay_tab:
    components.html(player_html, height=2150, scrolling=False)
    st.caption(
        f"Official Driver of the Day records: [F1DB](https://github.com/f1db/f1db) (CC BY 4.0); "
        f"award definition and results: [Formula 1]"
        f"(https://www.formula1.com/en/results/{year}/awards/driver-of-the-day)."
    )

with championship_tab:
    st.subheader(f"{year} FIA Formula One World Championship")
    st.caption("Season points after each completed Grand Prix, including sprint points where applicable.")
    championship_loaded_key = f"championship_loaded_{year}"
    if st.button("Load championship standings", type="primary", key=f"load_championship_{year}"):
        st.session_state[championship_loaded_key] = True
    if st.session_state.get(championship_loaded_key, False):
        try:
            snapshots = get_championship_data(year)
        except Exception as exc:
            st.error(f"Couldn't load championship points: {exc}")
            snapshots = []
        if not snapshots:
            st.warning("No championship results were available for this season.")
        else:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            round_numbers = pd.to_numeric(schedule.get("RoundNumber"), errors="coerce").dropna()
            scheduled_rounds = int(round_numbers.max()) if not round_numbers.empty else snapshots[-1]["round"]
            completed_through = snapshots[-1]["round"]
            if completed_through >= scheduled_rounds:
                st.caption(f"Final standings · {snapshots[-1]['event']} · Round {completed_through}")
            else:
                st.caption(f"Progressive standings through {snapshots[-1]['event']} · Round {completed_through} of {scheduled_rounds}")

            drivers_tab, constructors_tab = st.tabs(["Drivers' Championship", "Constructors' Championship"])
            with drivers_tab:
                final_drivers = snapshots[-1]["drivers"]
                driver_table = pd.DataFrame([
                    {"Pos": index + 1, "Driver": item["name"], "Team": item["team"], "Points": item["points"]}
                    for index, item in enumerate(final_drivers)
                ])
                st.dataframe(driver_table, use_container_width=True, hide_index=True)
                driver_names = [item["name"] for item in final_drivers]
                default_drivers = driver_names[:5]
                selected_drivers = st.multiselect("Drivers to track", driver_names, default=default_drivers, key=f"drivers_progress_{year}")
                driver_progress_rows = [
                    {"Round": f"R{snapshot['round']:02d} · {snapshot['event']}", "_round": snapshot["round"],
                     **{item["name"]: item["points"] for item in snapshot["drivers"]}}
                    for snapshot in snapshots
                ]
                driver_progress = pd.DataFrame(driver_progress_rows).sort_values("_round").set_index("Round")
                if selected_drivers:
                    st.line_chart(driver_progress.reindex(columns=selected_drivers).fillna(0), y=selected_drivers, height=360)
            with constructors_tab:
                final_constructors = snapshots[-1]["constructors"]
                constructor_table = pd.DataFrame([
                    {"Pos": index + 1, "Constructor": item["team"], "Points": item["points"]}
                    for index, item in enumerate(final_constructors)
                ])
                st.dataframe(constructor_table, use_container_width=True, hide_index=True)
                constructor_names = [item["team"] for item in final_constructors]
                default_constructors = constructor_names[:5]
                selected_constructors = st.multiselect("Constructors to track", constructor_names, default=default_constructors, key=f"constructors_progress_{year}")
                constructor_progress_rows = [
                    {"Round": f"R{snapshot['round']:02d} · {snapshot['event']}", "_round": snapshot["round"],
                     **{item["team"]: item["points"] for item in snapshot["constructors"]}}
                    for snapshot in snapshots
                ]
                constructor_progress = pd.DataFrame(constructor_progress_rows).sort_values("_round").set_index("Round")
                if selected_constructors:
                    st.line_chart(constructor_progress.reindex(columns=selected_constructors).fillna(0), y=selected_constructors, height=360)
