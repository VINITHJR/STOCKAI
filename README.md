# StockAI — Indian Stock Market Intelligence Platform

StockAI is an AI-powered Indian stock market analysis platform combining market data integration, technical analysis, machine-learning predictions, AI-driven signal intelligence, historical backtesting and validation dashboards.

It provides a unified interface for exploring market information, analysing model outputs and evaluating historical strategy behaviour.

> **Disclaimer:** StockAI is an educational and research software project. Its predictions and historical backtests do not guarantee future performance and do not constitute investment advice. Do not use the system as the sole basis for investment or trading decisions.

## Features

- Indian market monitoring: NIFTY 50, SENSEX, BANK NIFTY and India VIX.
- Upstox API integration for supported market data.
- Technical indicators and stock analysis.
- LSTM price prediction.
- Random Forest trend classification.
- AI signal intelligence and multi-model analysis.
- Market-regime and risk analysis.
- Historical backtesting and transaction-cost modelling.
- Walk-forward validation and data-integrity diagnostics.
- Phase 15 risk, drawdown, portfolio-stability and market-regime research dashboard.

Feature availability depends on configuration, data-provider access and compatible model artifacts.

## Technology Stack

| Area | Technologies |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Frontend | React, TypeScript, Vite |
| Machine Learning | TensorFlow, Keras, Scikit-learn |
| Data Processing | Pandas, NumPy, SciPy |
| Market Data | Upstox API |
| Testing | Pytest, Python regression scripts |
| API Documentation | FastAPI / OpenAPI |

## Repository Structure

```text
StockAI/
├── backend/
│   ├── app/
│   │   ├── advanced_ai/
│   │   ├── backtest/
│   │   ├── ml/
│   │   │   ├── models/
│   │   │   └── phase15/
│   │   │       ├── reference/
│   │   │       ├── reports/
│   │   │       ├── tests/
│   │   │       └── validation/
│   │   ├── services/
│   │   ├── startup.py
│   │   └── main.py
│   └── .env.example
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── package-lock.json
├── scripts/
├── tests/
├── requirements.txt
├── requirements-dev.txt
├── requirements-ml.txt
├── requirements-live.txt
├── .gitignore
└── README.md
```

## Tested Environment

### Backend

The backend dependency lock was installed and tested using:

- Python 3.11.17
- Linux under WSL
- TensorFlow 2.15.1
- Keras 2.15.0
- NumPy 1.26.4
- Pandas 3.0.6
- Scikit-learn 1.9.0
- FastAPI 0.143.3
- Uvicorn 0.54.0

### Frontend

- Node.js 24.x
- npm 11.x
- React 19
- Vite 8.3.5
- TypeScript 6

The backend dependency lock has been verified in the tested Linux/WSL environment. Native Windows and macOS backend installations have not been independently verified.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd StockAI
```

### 2. Create the Python environment

```bash
python -m venv .venv
```

Activate on Linux/WSL:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

For development and testing:

```bash
python -m pip install -r requirements-dev.txt
```

The `requirements-ml.txt` and `requirements-live.txt` files contain component-specific dependency subsets. They are not complete replacement environments.

### 3. Configure Upstox credentials

The environment template is located at:

```text
backend/.env.example
```

Create your local configuration:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` and provide credentials where required:

```env
UPSTOX_API_KEY=
UPSTOX_API_SECRET=
UPSTOX_REDIRECT_URI=
UPSTOX_ACCESS_TOKEN=
AUTO_TRAIN_MODELS=false
```

Never commit `backend/.env` or expose API credentials and access tokens.

Keep automatic model training disabled unless you intentionally initiate an authorised training workflow.

Without the required Upstox credentials, authenticated live-market features will not be available.

### 4. Install frontend dependencies

Open a separate terminal:

```bash
cd frontend
npm ci
```

### 5. Start the backend

From the repository root, with the Python environment activated:

```bash
python -m uvicorn backend.app.main:app --reload --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

OpenAPI specification:

```text
http://127.0.0.1:8000/openapi.json
```

### 6. Start the frontend

In another terminal:

```bash
cd frontend
npm run dev
```

Open the local URL printed by Vite, normally:

```text
http://localhost:5173
```

## Testing

### Backend automated tests

Run the selected Phase 15 and backend regression suite:

```bash
python -m pytest backend/app/ml/phase15/tests tests -v
```

Fundamental Intelligence tests:

```bash
export PYTHONPATH="$PWD/backend/app/advanced_ai:$PWD"

python -m pytest backend/app/advanced_ai/test_fundamental_intelligence.py -v
```

Additional synthetic regression scripts:

```bash
python backend/app/advanced_ai/test_stock_regime_intelligence.py

python backend/app/advanced_ai/test_technical_analysis_intelligence.py

python tests/test_phase8_layer8_synthetic.py
```

Phase 15 frontend integration check:

```bash
python scripts/test_phase15_frontend.py
```

Phase 15 HTTP smoke test requires the backend to be running on port 8000:

```bash
python scripts/test_phase15_report_api.py
```

Do not run real-data tests unless the required data-provider access and execution scope have been reviewed.

### Frontend checks

From the frontend directory:

```bash
cd frontend
npm ci
npm run lint
npx tsc -b
npx vite build
```

**Known frontend limitation:** The TypeScript project check currently reports one unused `riskClass` declaration in `src/pages/Phase14AdvancedAIPage.tsx`. The strict unused-variable checks have been retained. The Vite production bundle builds successfully independently. The frontend linter currently reports warnings but no errors.

The frontend dependency audit reported zero vulnerabilities after the November 4, 2026 dependency updates.

## Phase 15 Research Reports

The repository includes an approved research-only Phase 15 Part 4.3 report run.

- Run ID: `20261102_072010`
- Part: `4.3`
- Title: Risk, Drawdown, Portfolio Stability and Market-Regime Assessment

The report API exposes:

```text
GET /api/phase15/overview
GET /api/phase15/manifest
GET /api/phase15/reports/{report_key}
```

The approved manifest contains 16 reports.

Important research boundaries:

- `research_only` is true.
- The final holdout begins on 2026-04-17.
- The reported Part 4.3 run states that the final holdout was not used.
- The maximum Part 4.2.2 date is 2026-04-16.
- The reports describe previously reviewed out-of-fold predictions and historical simulations.
- The reliability state is retrospective research, not a live risk gate.

The tested portfolios did not demonstrate uniformly positive net returns. The reports contain negative net returns and substantial drawdowns under the evaluated assumptions. Review the complete metrics, sample sizes and limitations before interpreting the results.

Historical performance is not evidence of future profitability.

## Model Artifact and Provenance Notes

Successful model loading and synthetic inference do not establish:

- Complete training-data provenance.
- Absence of training or validation leakage.
- Independent out-of-sample predictive performance.
- Suitability for live trading.

The training-data provenance of the bundled model artifacts has not been fully established for this release. In particular, do not represent the Random Forest artifact as independently verified or leakage-free.

Do not retrain, replace or overwrite production model artifacts without a separately reviewed and authorised workflow.

Never load model or pickle files from untrusted sources.

## Security

- Store credentials in local environment files.
- Never commit `.env`, private keys or access tokens.
- Do not publish personal account information.
- Review third-party dependency advisories regularly.
- Do not load untrusted pickle artifacts.
- Respect the data provider's API terms and rate limits.
- Keep research-only and holdout data boundaries intact.

## Project Status

StockAI is an actively developed research and software project.

The current release preparation has included selected backend regression tests, model-load and synthetic-inference checks, frontend production bundling, dependency-security checks and Phase 15 report API validation.

These checks do not constitute certification for live trading, financial advice or guaranteed predictive performance.

## License

StockAI is released under the MIT License.

Copyright (c) 2026 Vinith J R

See the [LICENSE](LICENSE) file for the complete license text.
