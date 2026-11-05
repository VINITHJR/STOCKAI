# StockAI Backend

## Run from the StockAI project root

```powershell
python -m uvicorn backend.app.main:app --reload
```

Do not run `app.main:app` from inside `backend` because the current backend uses the `backend.app.*` import convention.

## Environment
Copy `.env.example` to `backend/.env` and add your existing Upstox credentials/access token.

## Phase 8
The `app/backtest/` package contains the Historical Backtest Engine foundation.

## Services restored/compatibility layer
`app/services/` contains the Upstox, market-stream, fundamentals, news/events, prediction, and trend service modules required by `app/main.py`.
