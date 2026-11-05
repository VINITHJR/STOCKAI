# StockAI Frontend

The StockAI frontend is a React, TypeScript and Vite application that provides an interactive dashboard for Indian stock-market monitoring, AI-powered analysis, portfolio insights, backtesting and validation reports.

## Technology Stack

- React 19
- TypeScript 6
- Vite 8
- Tailwind CSS 4
- React Router
- Recharts
- Lightweight Charts
- Lucide React

## Requirements

The frontend has been tested with:

- Node.js 24.x
- npm 11.x

## Installation

Open a terminal in the `frontend` directory.

Install the exact dependency versions from the lockfile:

```bash
npm ci
```

## Development Server

Start the frontend development server:

```bash
npm run dev
```

Vite will display the local development URL, normally:

```text
http://localhost:5173
```

To expose the development server on your local network:

```bash
npm run dev -- --host 0.0.0.0
```

The FastAPI backend must be started separately from the repository root.

Refer to the root `README.md` for backend installation, Upstox configuration and application startup instructions.

## Production Build

Build the frontend production bundle:

```bash
npx vite build
```

The generated production files are placed in:

```text
frontend/dist/
```

Preview the production bundle:

```bash
npx vite preview
```

The production bundle has been successfully tested with Vite 8.3.5.

A chunk-size warning remains because the main JavaScript bundle exceeds 500 KB. Code splitting can be considered as a future performance improvement.

## TypeScript Validation

Run:

```bash
npx tsc -b
```

**Known issue:** The current strict TypeScript check reports one unused `riskClass` declaration in:

```text
src/pages/Phase14AdvancedAIPage.tsx
```

The strict unused-variable compiler settings have intentionally been retained. The protected Phase 14 source file has not been modified to suppress this diagnostic.

## Code Quality

Run the frontend linter:

```bash
npm run lint
```

The latest lint run reported warnings but no errors. Review React hook dependencies, effect state updates and ref-access warnings before changing application behaviour.

## Dependency Security

Check installed dependencies for known security vulnerabilities:

```bash
npm audit
```

The frontend dependencies were updated during release preparation.

The full npm audit and production-only npm audit both reported zero vulnerabilities after the November 4, 2026 updates.

For reproducible dependency installation, use:

```bash
npm ci
```

Avoid applying forced major-version dependency upgrades without reviewing their compatibility.

## Project Structure

```text
frontend/
├── public/
│   ├── data/
│   ├── images/
│   └── assets/
├── src/
│   ├── components/
│   ├── pages/
│   ├── services/
│   ├── App.tsx
│   └── main.tsx
├── package.json
├── package-lock.json
├── tsconfig.json
└── vite.config.ts
```

## Backend Connection

The frontend communicates with the StockAI FastAPI backend.

Default local backend address:

```text
http://127.0.0.1:8000
```

FastAPI interactive documentation:

```text
http://127.0.0.1:8000/docs
```

Start the backend separately:

```bash
cd ..
python -m uvicorn backend.app.main:app --reload --port 8000
```

Live market-data functionality may require valid Upstox API credentials configured in `backend/.env`.

## Phase 15 Research Dashboard

The frontend includes the Phase 15 Risk and Stability dashboard.

It presents research reports covering:

- Portfolio risk metrics
- Daily risk and equity series
- Drawdown episodes
- Fold-level risk and trade attribution
- Market-regime exposure
- Stock-level risk concentration
- Prediction reliability
- Source-metric reconciliation

The dashboard consumes the Phase 15 read-only report API.

These reports are research outputs, not live trading instructions or guarantees of future profitability.

## Security Notes

- Never put API credentials in frontend source code.
- Do not expose Upstox access tokens to the browser.
- Keep backend secrets in local environment files.
- Do not commit `.env` files.
- Review dependency security advisories regularly.

## Additional Documentation

For complete application architecture, backend setup, API configuration, testing instructions and research limitations, see the root `README.md`.