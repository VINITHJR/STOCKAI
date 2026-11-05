from pathlib import Path
service=Path("frontend/src/services/phase15Api.ts").read_text(encoding="utf-8")
page=Path("frontend/src/pages/Phase15RiskStabilityPage.tsx").read_text(encoding="utf-8")
app=Path("frontend/src/App.tsx").read_text(encoding="utf-8")
assert "/overview" in service and "/reports/" in service
for key in ("daily_risk_series","drawdown_episodes","market_regime_exposure","fold_calendar_risk","stock_risk_concentration","reliability_exposure","source_metric_reconciliation"):
    assert key in page, key
assert 'path="/phase15"' in app and 'href="/phase15"' in app
assert "Phase15RiskStabilityPage" in app
print("PASS: frontend service and dashboard files present; Phase 15 route/navigation registered.")
print("Next: cd frontend && npm run build")
