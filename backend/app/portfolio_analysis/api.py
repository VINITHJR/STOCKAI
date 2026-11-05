"""
Phase 13 - Portfolio Analysis API
"""

from fastapi import APIRouter

from .routes import router


api_router = APIRouter()

api_router.include_router(
    router
)