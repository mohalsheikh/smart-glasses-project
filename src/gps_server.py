# src/gps_server.py

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.utils import config


app = FastAPI()

# ✅ Allow requests from anywhere (phone browser via tunnel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOCATION_PATH = Path(config.NAV_LOCATION_FILE)
LOCATION_PATH.parent.mkdir(parents=True, exist_ok=True)


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>📍 GPS Feed</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: -apple-system, system-ui, sans-serif; padding: 18px; }
    button { font-size: 16px; padding: 10px 14px; }
    pre { background: #111; color: #0f0; padding: 12px; border-radius: 10px; overflow: auto; }
    .muted { color: #666; }
  </style>
</head>
<body>
  <h2>📍 GPS Feed</h2>
  <p class="muted">This page sends your location to the server continuously.</p>

  <button id="start">Start sending location</button>
  <button id="stop" disabled>Stop</button>

  <h3>Status</h3>
  <div id="status">Idle</div>

  <h3>Last payload</h3>
  <pre id="out">—</pre>

  <script>
    const out = document.getElementById("out");
    const status = document.getElementById("status");
    const btnStart = document.getElementById("start");
    const btnStop = document.getElementById("stop");

    let watchId = null;

    function setStatus(s) { status.textContent = s; }

    async function sendLocation(lat, lon, acc) {
      const payload = { lat, lon, accuracy_m: acc, t: Date.now() };
      out.textContent = "Sending: " + JSON.stringify(payload, null, 2);

      const resp = await fetch("/api/location", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error("Server error: " + resp.status + " " + txt);
      }
    }

    btnStart.onclick = async () => {
      if (!navigator.geolocation) {
        setStatus("Geolocation not supported in this browser.");
        return;
      }

      btnStart.disabled = true;
      btnStop.disabled = false;

      setStatus("Requesting location permission...");

      watchId = navigator.geolocation.watchPosition(
        async (pos) => {
          try {
            setStatus("Got GPS fix. Sending to server...");
            await sendLocation(pos.coords.latitude, pos.coords.longitude, pos.coords.accuracy);
            setStatus("✅ Sent. (watching...)");
          } catch (e) {
            setStatus("❌ " + e.message);
          }
        },
        (err) => {
          setStatus("❌ GPS error: " + err.message);
        },
        {
          enableHighAccuracy: true,
          maximumAge: 1000,
          timeout: 10000
        }
      );
    };

    btnStop.onclick = () => {
      if (watchId !== null) navigator.geolocation.clearWatch(watchId);
      watchId = null;
      btnStart.disabled = false;
      btnStop.disabled = true;
      setStatus("Stopped.");
    };
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(HTML)


@app.post("/api/location")
async def location(req: Request) -> JSONResponse:
    payload: Dict[str, Any] = await req.json()

    lat = float(payload.get("lat"))
    lon = float(payload.get("lon"))
    acc = float(payload.get("accuracy_m", 9999.0))
    client_t = payload.get("t", None)

    record = {
        "lat": lat,
        "lon": lon,
        "accuracy_m": acc,
        "client_t": client_t,
        "server_t": time.time(),
    }

    LOCATION_PATH.write_text(json.dumps(record, indent=2))

    if config.DEBUG:
        print(f"📍 GPS update saved -> {LOCATION_PATH} | {record}")

    return JSONResponse({"ok": True, "saved_to": str(LOCATION_PATH)})
