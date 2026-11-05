# Phase 14 — Correct Phase 14.1 / 14.2 Backend Integration

## Files

Copy these files into `backend/app/advanced_ai/`:

- `routes.py`
- `orchestrator.py`
- `ai_decision_intelligence.py`
- `ai_confidence_reliability.py`
- `final_ai_intelligence_aggregator.py`

## Integration contract

### Phase 14.1 — Technical Intelligence

- Uses the same stock-specific daily OHLCV candles already fetched for Phase 9 market-regime analysis.
- Runs `TechnicalAnalysisIntelligence.analyze(candles)` exactly once.
- Its `signal` and `confidence` are passed to Phase 14 Layer 1.
- Layer 1 weight: **7%**.
- It is also included in Layer 2 supporting reliability and Layer 3 explanations through the existing `technical_signal` path.

### Phase 14.2 — Stock Regime Intelligence

- Uses the same stock-specific candles.
- Runs independently through `StockRegimeIntelligence`.
- It is returned under `phase14_input_source.context.stock_regime_intelligence` and the explicit `phase14_2` trace block.
- It is intentionally **context/diagnostic only** and has **0% duplicate Layer 1 weight**. This avoids double-counting the same OHLCV evidence through another directional weight.

### Layer 1 weights

- AI signal: 40%
- Ensemble: 25%
- Market: 20%
- Technical 14.1: 7%
- Fundamental 14.3: 5%
- News: 3%

Total = 100%.

AI/ensemble conflict penalties affect only the correlated AI/ensemble bucket. Independent market, technical, fundamental and news evidence is not penalized by model conflict.

## Validation

Compile check:

```powershell
python -m compileall backend/app/advanced_ai
```

Endpoint test:

```powershell
$body = @{ instrument_key = "NSE_EQ|INE009A01021" } | ConvertTo-Json
$r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/advanced-ai/run" -Method POST -ContentType "application/json" -Body $body
$r.result.phase14_input_source.context.technical_intelligence
$r.result.phase14_input_source.context.stock_regime_intelligence
$r.result.phase14_input_source.phase14_1
$r.result.phase14_input_source.phase14_2
```

The response should show real Phase 14.1 and 14.2 values for the selected instrument and should not reuse another stock's values.
