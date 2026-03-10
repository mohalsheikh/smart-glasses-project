# src/location_server.py

from __future__ import annotations

import os
import json
import time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.utils import config

app = FastAPI()


class Loc(BaseModel):
    lat: float
    lon: float
    accuracy_m: float | None = None


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Smart Glasses GPS Feed</title>
  <style>
    body { font-family: system-ui; padding: 16px; }
    button { font-size: 18px; padding: 12px 16px; }
    pre { background: #f5f5f5; padding: 12px; border-radius: 8px; }
  </style>
</head>
<body>
  <h2>📍 GPS Feed</h2>
  <button onclick="start()">Start sending location</button>
  <pre id="out">Idle</pre>

<script>
async function start(){
  const out = document.getElementById('out');
  if (!navigator.geolocation) {
    out.textContent = "Geolocation not supported.";
    return;
  }

  navigator.geolocation.watchPosition(async (pos) => {
    const payload = {
      lat: pos.coords.latitude,
      lon: pos.coords.longitude,
      accuracy_m: pos.coords.accuracy
    };
    out.textContent = "Sending: " + JSON.stringify(payload, null, 2);

    await fetch('/location', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
  }, (err) => {
    out.textContent = "Error: " + err.message;
  }, { enableHighAccuracy: true, maximumAge: 1000, timeout: 10000 });
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.post("/location")
def location(loc: Loc):
    os.makedirs(os.path.dirname(config.NAV_LOCATION_FILE), exist_ok=True)
    data = {
        "lat": loc.lat,
        "lon": loc.lon,
        "accuracy_m": loc.accuracy_m,
        "ts": time.time(),
    }
    with open(config.NAV_LOCATION_FILE, "w") as f:
        json.dump(data, f)
    return {"ok": True}
