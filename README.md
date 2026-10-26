# Trackside Replay

Trackside Replay is a Streamlit dashboard for replaying historical Formula 1 sessions. It combines an animated circuit map, timing tower, driver details, and recorded telemetry.

## Features

- Replay recorded race, qualifying, and practice sessions from 2018 onward.
- Scrub and play sessions at adjustable speeds, with car positions animated in the browser.
- View timing, gaps, lap times, tyre compounds and ages, purple sectors, and session highlights.
- Select a driver to inspect their recorded telemetry and lap details.
- View season championship standings and cumulative points.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

The first load of a session may take a few minutes while FastF1 downloads and caches historical data. Later loads use the local cache. Data availability depends on the session and FastF1's upstream sources.

This project uses historical timing data through FastF1. It does not provide live F1 timing or broadcast video.
