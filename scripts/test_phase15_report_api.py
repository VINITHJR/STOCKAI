#!/usr/bin/env python3
"""HTTP smoke test. Start StockAI's backend before running this script."""
import json
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

BASE = "http://127.0.0.1:8000/api/phase15"
RUN = "20261102_072010"

def get(path):
    try:
        with urlopen(BASE + path, timeout=15) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"{path}: HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    except URLError as exc:
        raise RuntimeError("Cannot reach backend; start Uvicorn first.") from exc

status, overview = get("/overview")
assert status == 200 and overview["status"] == "success"
assert overview["run_id"] == RUN
assert overview["portfolio_risk_metrics"]
assert overview["integrity"]["models_fitted"] is False
assert overview["integrity"]["upstox_api_called"] is False

status, manifest = get("/manifest")
assert status == 200 and manifest["run_id"] == RUN
assert len(manifest["reports"]) == 16, len(manifest["reports"])

for key in ("portfolio_risk_metrics", "daily_risk_series", "summary"):
    status, payload = get("/reports/" + key)
    assert status == 200 and payload["status"] == "success"

try:
    get("/reports/../../main")
except RuntimeError as exc:
    assert "HTTP 404" in str(exc) or "HTTP 422" in str(exc), str(exc)
else:
    raise AssertionError("Unexpectedly accepted an unknown/path-like report key.")

print("PASS: overview, manifest, three reports and unknown-key rejection.")
print("PASS: run_id", RUN, "| allowlisted reports", len(manifest["reports"]))
print("All requests were GET-only; no research or model code was executed.")
