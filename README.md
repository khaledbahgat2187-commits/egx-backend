# EGX Signals API

FastAPI backend that computes technical trading signals for Egyptian Exchange (EGX) stocks using yfinance data.

## Endpoints
- `GET /` — API info
- `GET /health` — status check
- `GET /signals` — scored signal list for all tickers
- `GET /signals/{ticker}` — detail for one ticker

## Deploy to Render
1. Connect this repo
2. Render detects `render.yaml` automatically
3. First build takes ~5 minutes (installs pandas, numpy, yfinance)
