"""Phase 8 backtest router aggregator.

Keep the existing Layer 1 endpoints and register the implemented
Phase 8 Layers 4–8 endpoints under their own existing prefixes.
Phase 10 validation routes are registered separately by main.py and are
intentionally not duplicated here.
"""

from fastapi import APIRouter

from backend.app.backtest.routes import router as layer1_router
from backend.app.backtest.layer4 import router as ml_prediction_router
from backend.app.backtest.layer5 import router as portfolio_router
from backend.app.backtest.layer6 import router as risk_router
from backend.app.backtest.layer7 import router as benchmark_router
from backend.app.backtest.layer8 import router as robustness_router

router = APIRouter()

router.include_router(layer1_router)
router.include_router(ml_prediction_router)
router.include_router(portfolio_router)
router.include_router(risk_router)
router.include_router(benchmark_router)
router.include_router(robustness_router)

__all__ = ["router"]
