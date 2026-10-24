# 🏎️ F1 Pit Wall

An animated **Formula 1 telemetry dashboard**. It replays a real, completed F1
session on a 2D track map: every driver is a coloured dot (official team colour
+ 3-letter abbreviation like `VER` / `HAM` / `LEC`) moving around the circuit,
with **play / pause / scrub-through-time** controls. Select a driver to open a
stats panel with lap times, position change, tire info, a per-stint tire
degradation estimate, and (stretch) telemetry traces with a two-driver overlay.

## The problem it solves

Formula 1 publishes a huge amount of timing and telemetry data, but it is
locked away in raw tables that mean little to most fans. Pit Wall turns that
raw public data into an **intuitive visual tool** for fans and aspiring
motorsport engineers: you *watch* the race unfold on a map and immediately see
who gained places, who is managing tires, and how two drivers compare on a lap.

## Features

- **Animated 2D track map** built from real X/Y position data, with team-coloured
  dots and driver abbreviations, play/pause button, and a time scrubber.
- **Driver stats panel**: current position, grid, position change since the
  start, current tire compound + age, best/last lap, and a lap-time chart.
- **Tire degradation** estimate per stint (lap-time trend, in seconds/lap).
- **Telemetry (stretch)**: speed / throttle / brake / gear / DRS traces for the
  selected driver's fastest lap, plus a two-driver comparison overlay.
- **Works for any session** (any year/event/session type), not hardcoded to one
  race, with a sensible, reliable default.

## Tech stack

- **Python 3.11**
- **[FastF1](https://docs.fastf1.dev)** for lap timing, car telemetry, X/Y
  position, tire and results data (Replay mode).
- **[OpenF1](https://openf1.org)** (`https://api.openf1.org/v1`), a free,
  no-auth REST API, for **near-live** car positions and standings (Live mode).
  Accessed with the Python standard library (`urllib`), so it adds **no new
  dependency**.
- **Streamlit + Plotly** for the animated, interactive UI.

## Live mode (OpenF1)

The sidebar has a **Mode** toggle:

- **Replay (FastF1)** — the original behaviour: pick a completed session and
  play it back with full stats, tire degradation and telemetry.
- **Live (OpenF1)** — follows the **current or most recent** session in
  near-real-time. It polls OpenF1 every ~3 seconds for the latest car position
  per driver and moves the dots on a track outline that fills in as the cars
  run. Best-effort live **standings and gaps** are shown alongside the map.

What to know about Live mode:

- **It is near-live, not instant.** OpenF1 is roughly **3 seconds behind** the
  track. For this project that delay is fine.
- **It only shows data while a session is actually running ("green").** OpenF1
  streams car locations only when cars are on track. If **no session is live**,
  Live mode shows a clear message (not a blank screen) and keeps polling so it
  lights up the instant a session starts.
- **Demo guidance:** if you are demoing when **no session is live** (e.g. a
  quiet Sunday morning), use **Replay (FastF1)** to play back a completed race.
  Switch to Live during an actual session.
- Detailed lap/tire/telemetry stats need FastF1 and are therefore **Replay-only**.
  In Live mode the panel degrades gracefully to driver identity + live
  standings rather than crashing.

Live mode needs a working internet connection and does not use the FastF1
cache. It targets the latest session via OpenF1's `session_key=latest`.

## Install

```bash
# from the f1-pitwall/ directory
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## ⚠️ Populate the cache FIRST (read this before your demo)

FastF1 downloads session data from the F1 servers the first time you open a
session (**50–100 MB** per session), then caches it in a local `./cache`
directory. Subsequent runs are fast and work even if the network is flaky.

> **Run the app once, early, on a good connection and load the session(s) you
> plan to demo.** After that the data is cached locally and you are immune to
> network issues during the demo.

You can pre-warm the cache without the UI:

```bash
python3 -c "import fastf1; fastf1.Cache.enable_cache('cache'); \
s = fastf1.get_session(2024, 'United States', 'R'); s.load()"
```

### Known issue: HTTP 401/403 on recent sessions

Since around mid-October 2026, F1's live-timing server can return **HTTP
401/403** to unauthenticated requests on a *cold* cache
([FastF1 issue #978](https://github.com/theOehrly/Fast-F1/issues/978)). This
mainly affects recent **2025/2026** sessions. Pit Wall handles this gracefully:
it shows a clear message (not a stack trace) explaining the likely cause and
suggesting a cached or 2024 session. The **default demo session is the 2024
United States GP (COTA)** precisely because 2024 data is reliable.

## Run

```bash
streamlit run src/app.py
```

Then in the sidebar: pick a **Year**, **Event** and **Session**, click **Load
session**, choose a **Driver**, and press **▶ Play** on the map.

## Project layout

```
f1-pitwall/
├── README.md
├── requirements.txt
├── .gitignore               # ignores cache/, __pycache__, venv
├── src/
│   ├── app.py               # Streamlit UI (map, scrubber, driver panel)
│   ├── data/
│   │   ├── loader.py        # swappable data source + FastF1 implementation
│   │   ├── openf1_source.py # OpenF1 near-live data source (same interface)
│   │   └── analysis.py      # lap times, position change, tire degradation
│   └── viz/
│       └── track_map.py     # circuit outline + animated car dots (Plotly)
└── tests/
    └── test_offline.py      # network-free unit tests
```

## Swappable data layer & future OpenF1 live mode

The UI never talks to a data provider directly. It depends only on the abstract
`SessionDataSource` interface in `src/data/loader.py`, which returns plain
dataclasses (`SessionData`, `DriverInfo`). There are two implementations:

- `FastF1DataSource` (historical replay), via `default_source()`.
- `OpenF1DataSource` (near-live), in `src/data/openf1_source.py`, via
  `live_source()`.

Because both return the **same** `SessionData`, the animation (`track_map.py`)
and the stats panel work against either source without structural changes. The
OpenF1 source builds the shared time axis, resamples each driver's X/Y onto it,
derives the track outline from the densest location trace, and maps
`driver_number -> name_acronym` abbreviations and `#`-prefixed `team_colour`
exactly the way the FastF1 source does.

## Development

```bash
ruff check src tests      # lint
black src tests           # format
pytest                    # offline unit tests (no network needed)
```
