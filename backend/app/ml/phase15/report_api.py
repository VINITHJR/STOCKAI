"""Read-only API for the approved StockAI Phase 15 Part 4.3 reports."""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/phase15", tags=["Phase 15 Reports"])
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
EXPECTED_RUN_ID = "20261102_072010"
SUMMARY_FILENAME = f"phase15_part4_3_{EXPECTED_RUN_ID}_summary.json"

# Static allowlist. API callers provide a key, never a filename or filesystem path.
REPORT_FILES: dict[str, str] = {
    "portfolio_risk_metrics": f"phase15_part4_3_{EXPECTED_RUN_ID}_portfolio_risk_metrics.csv",
    "daily_risk_series": f"phase15_part4_3_{EXPECTED_RUN_ID}_daily_risk_series.csv",
    "drawdown_episodes": f"phase15_part4_3_{EXPECTED_RUN_ID}_drawdown_episodes.csv",
    "fixed_quantity_anchor_reconciliation": f"phase15_part4_3_{EXPECTED_RUN_ID}_fixed_quantity_anchor_reconciliation.csv",
    "fold_calendar_risk": f"phase15_part4_3_{EXPECTED_RUN_ID}_fold_calendar_risk.csv",
    "fold_trade_attribution": f"phase15_part4_3_{EXPECTED_RUN_ID}_fold_trade_attribution.csv",
    "market_regime_calendar": f"phase15_part4_3_{EXPECTED_RUN_ID}_market_regime_calendar.csv",
    "market_regime_exposure": f"phase15_part4_3_{EXPECTED_RUN_ID}_market_regime_exposure.csv",
    "market_regime_trade_performance": f"phase15_part4_3_{EXPECTED_RUN_ID}_market_regime_trade_performance.csv",
    "prediction_reliability_history": f"phase15_part4_3_{EXPECTED_RUN_ID}_prediction_reliability_history.csv",
    "reliability_exposure": f"phase15_part4_3_{EXPECTED_RUN_ID}_reliability_exposure.csv",
    "reliability_exposure_comparison": f"phase15_part4_3_{EXPECTED_RUN_ID}_reliability_exposure_comparison.csv",
    "source_metric_reconciliation": f"phase15_part4_3_{EXPECTED_RUN_ID}_source_metric_reconciliation.csv",
    "stock_risk_concentration": f"phase15_part4_3_{EXPECTED_RUN_ID}_stock_risk_concentration.csv",
    "vix_thresholds": f"phase15_part4_3_{EXPECTED_RUN_ID}_vix_thresholds.csv",
    "summary": SUMMARY_FILENAME,
}
MAX_REPORT_ROWS = 20_000


def _report_path(report_key: str) -> Path:
    filename = REPORT_FILES.get(report_key)
    if filename is None:
        raise HTTPException(status_code=404, detail="Unknown Phase 15 report key.")
    root = REPORTS_DIR.resolve()
    path = (root / filename).resolve()
    if path.parent != root:
        raise HTTPException(status_code=500, detail="Invalid report path configuration.")
    if not path.is_file():
        raise HTTPException(status_code=503, detail=f"Saved report is missing: {filename}")
    return path


def _coerce(value: str | None) -> Any:
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if re.fullmatch(r"[+-]?\d+", value):
        try:
            return int(value)
        except ValueError:
            return value
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except ValueError:
        return value


def _read_csv(report_key: str) -> dict[str, Any]:
    path = _report_path(report_key)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            rows = []
            for index, row in enumerate(reader, 1):
                if index > MAX_REPORT_ROWS:
                    raise HTTPException(status_code=413, detail="Report exceeds API row limit.")
                rows.append({k: _coerce(v) for k, v in row.items() if k is not None})
        return {
            "report_key": report_key,
            "source_file": path.name,
            "columns": columns,
            "record_count": len(rows),
            "rows": rows,
        }
    except HTTPException:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise HTTPException(status_code=500, detail="Unable to read saved report.") from exc


def _read_summary() -> dict[str, Any]:
    path = _report_path("summary")
    try:
        with path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Unable to read saved summary.") from exc
    if not isinstance(summary, dict) or summary.get("run_id") != EXPECTED_RUN_ID:
        raise HTTPException(status_code=409, detail="Summary run ID does not match approved run.")
    return summary


@router.get("/overview")
def get_phase15_overview() -> dict[str, Any]:
    summary = _read_summary()
    metrics = _read_csv("portfolio_risk_metrics")
    return {
        "status": "success",
        "phase": 15,
        "part": "4.3",
        "run_id": EXPECTED_RUN_ID,
        "title": summary.get("title"),
        "research_only": summary.get("research_only", True),
        "source_runs": summary.get("source_runs", {}),
        "portfolios": summary.get("portfolios", []),
        "slippage_scenarios_bps_per_side": summary.get("slippage_scenarios_bps_per_side", []),
        "holdout": summary.get("holdout", {}),
        "integrity": {
            key: summary.get(key) for key in (
                "models_fitted", "predictions_modified", "hyperparameters_changed",
                "upstox_api_called", "source_reports_modified",
                "protected_artifacts_modified", "source_fingerprints_unchanged",
            )
        } | {
            "part4_2_2_trade_source_reconciliation":
                summary.get("part4_2_2_trade_source_reconciliation", {})
        },
        "limitations": summary.get("limitations", []),
        "portfolio_risk_metrics": metrics["rows"],
    }


@router.get("/manifest")
def get_phase15_manifest() -> dict[str, Any]:
    reports = []
    for key, filename in REPORT_FILES.items():
        path = _report_path(key)
        if key == "summary":
            reports.append({
                "report_key": key, "source_file": filename, "format": "json",
                "record_count": 1, "columns": [], "size_bytes": path.stat().st_size,
            })
        else:
            data = _read_csv(key)
            reports.append({
                "report_key": key, "source_file": filename, "format": "csv",
                "record_count": data["record_count"], "columns": data["columns"],
                "size_bytes": path.stat().st_size,
            })
    return {"status": "success", "phase": 15, "part": "4.3",
            "run_id": EXPECTED_RUN_ID, "reports": reports}


@router.get("/reports/{report_key}")
def get_phase15_report(report_key: str) -> dict[str, Any]:
    if report_key == "summary":
        return {"status": "success", "report_key": "summary",
                "source_file": SUMMARY_FILENAME, "data": _read_summary()}
    return {"status": "success", **_read_csv(report_key)}
