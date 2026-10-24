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
  position, tire and results data.
- **Streamlit + Plotly** for the animated, interactive UI.

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
│   │   └── analysis.py      # lap times, position change, tire degradation
│   └── viz/
│       └── track_map.py     # circuit outline + animated car dots (Plotly)
└── tests/
    └── test_offline.py      # network-free unit tests
```

## Swappable data layer & future OpenF1 live mode

The UI never talks to FastF1 directly. It depends only on the abstract
`SessionDataSource` interface in `src/data/loader.py`, which returns plain
dataclasses (`SessionData`, `DriverInfo`). Today the only implementation is
`FastF1DataSource` (historical replay). A future **OpenF1** near-live source can
implement the same interface and be dropped in via `default_source()` without
changing any UI or visualisation code — the animation, stats panel and
telemetry views would then work against near-live data unchanged.

## Development

```bash
ruff check src tests      # lint
black src tests           # format
pytest                    # offline unit tests (no network needed)
```
