# ============================================================
# PHASE 14 — ADVANCED AI
# API ROUTER
# ============================================================

from fastapi import APIRouter

from .routes import router


# ============================================================
# PUBLIC API ROUTER
# ============================================================

api_router = APIRouter(
    prefix="/api",
)

api_router.include_router(router)