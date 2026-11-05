from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .engine import PaperTradingEngine
from .execution import PaperExecutionEngine
from .metrics import PaperTradingMetricsCalculator
from .models import (
    OrderStatus,
    PaperOrder,
    PaperTradingConfig,
    PaperTrade,
    TradingMode,
)
from .orders import PaperOrderManager
from .portfolio import PaperPortfolio
from .trade_history import PaperTradeHistory


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/paper-trading",
    tags=["Phase 11 - Paper Trading"],
)


# ============================================================
# IN-MEMORY PHASE 11 STATE
# ============================================================

engine = PaperTradingEngine()
order_manager = PaperOrderManager()
metrics_calculator = PaperTradingMetricsCalculator()

portfolios: dict[str, PaperPortfolio] = {}
execution_engines: dict[str, PaperExecutionEngine] = {}
trade_histories: dict[str, PaperTradeHistory] = {}


# ============================================================
# REQUEST MODELS
# ============================================================

class CreateSessionRequest(BaseModel):
    instrument_key: str
    initial_capital: float = Field(
        default=100000.0,
        gt=0,
    )
    commission_percent: float = Field(
        default=0.0,
        ge=0,
    )
    slippage_percent: float = Field(
        default=0.0,
        ge=0,
    )
    max_position_value_percent: float = Field(
        default=100.0,
        gt=0,
        le=100,
    )


class CreateOrderRequest(BaseModel):
    signal: str
    confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )
    current_price: float = Field(
        gt=0,
    )
    quantity: float = Field(
        gt=0,
    )


class ExecuteOrderRequest(BaseModel):
    market_price: Optional[float] = Field(
        default=None,
        gt=0,
    )


class UpdatePriceRequest(BaseModel):
    current_price: float = Field(
        gt=0,
    )


# ============================================================
# SERIALIZATION HELPERS
# ============================================================

def _session_response(session) -> dict:
    return {
        "session_id": session.session_id,
        "instrument_key": session.config.instrument_key,
        "status": session.status.value,
        "initial_capital": session.config.initial_capital,
        "commission_percent": (
            session.config.commission_percent
        ),
        "slippage_percent": (
            session.config.slippage_percent
        ),
        "total_orders": session.total_orders,
        "total_trades": session.total_trades,
        "created_at": session.created_at,
        "started_at": session.started_at,
        "stopped_at": session.stopped_at,
    }


def _order_response(order: PaperOrder) -> dict:
    return {
        "order_id": order.order_id,
        "session_id": order.session_id,
        "instrument_key": order.instrument_key,
        "side": order.side.value,
        "order_type": order.order_type.value,
        "quantity": order.quantity,
        "requested_price": order.requested_price,
        "status": order.status.value,
        "filled_price": order.filled_price,
        "filled_quantity": order.filled_quantity,
        "created_at": order.created_at,
        "filled_at": order.filled_at,
        "rejection_reason": order.rejection_reason,
    }


def _position_response(position) -> dict:
    return {
        "instrument_key": position.instrument_key,
        "quantity": position.quantity,
        "average_entry_price": (
            position.average_entry_price
        ),
        "current_price": position.current_price,
        "market_value": position.market_value,
        "invested_value": position.invested_value,
        "realized_pnl": position.realized_pnl,
        "unrealized_pnl": position.unrealized_pnl,
        "opened_at": position.opened_at,
        "updated_at": position.updated_at,
    }


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def paper_trading_health() -> dict:
    """
    Phase 11 paper-trading API health check.
    """

    return {
        "status": "success",
        "phase": 11,
        "module": "paper_trading",
        "layers": 7,
        "message": "Paper trading API is ready",
    }


# ============================================================
# CREATE SESSION
# ============================================================

@router.post("/sessions")
def create_session(
    request: CreateSessionRequest,
) -> dict:

    try:
        config = PaperTradingConfig(
            instrument_key=request.instrument_key,
            initial_capital=request.initial_capital,
            trading_mode=TradingMode.PAPER,
            commission_percent=request.commission_percent,
            slippage_percent=request.slippage_percent,
            max_position_value_percent=(
                request.max_position_value_percent
            ),
        )

        session = engine.create_session(config)

        portfolios[session.session_id] = (
            PaperPortfolio(config)
        )

        execution_engines[session.session_id] = (
            PaperExecutionEngine(config)
        )

        trade_histories[session.session_id] = (
            PaperTradeHistory()
        )

        return {
            "status": "success",
            "session": _session_response(session),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# GET SESSION
# ============================================================

@router.get("/sessions/{session_id}")
def get_session(
    session_id: str,
) -> dict:

    try:
        session = engine.get_session(session_id)

        return {
            "status": "success",
            "session": _session_response(session),
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


# ============================================================
# LIST SESSIONS
# ============================================================

@router.get("/sessions")
def list_sessions() -> dict:

    return {
        "status": "success",
        "count": len(engine.list_sessions()),
        "sessions": [
            _session_response(session)
            for session in engine.list_sessions()
        ],
    }


# ============================================================
# START SESSION
# ============================================================

@router.post("/sessions/{session_id}/start")
def start_session(
    session_id: str,
) -> dict:

    try:
        session = engine.start_session(session_id)

        return {
            "status": "success",
            "session": _session_response(session),
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# PAUSE SESSION
# ============================================================

@router.post("/sessions/{session_id}/pause")
def pause_session(
    session_id: str,
) -> dict:

    try:
        session = engine.pause_session(session_id)

        return {
            "status": "success",
            "session": _session_response(session),
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# STOP SESSION
# ============================================================

@router.post("/sessions/{session_id}/stop")
def stop_session(
    session_id: str,
) -> dict:

    try:
        session = engine.stop_session(session_id)

        return {
            "status": "success",
            "session": _session_response(session),
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# CREATE ORDER FROM AI SIGNAL
# ============================================================

@router.post("/sessions/{session_id}/orders")
def create_order(
    session_id: str,
    request: CreateOrderRequest,
) -> dict:

    try:
        order = engine.create_order_from_signal(
            session_id=session_id,
            signal=request.signal,
            confidence=request.confidence,
            current_price=request.current_price,
            quantity=request.quantity,
        )

        # Neutral / hold signal
        if order is None:
            return {
                "status": "success",
                "order_created": False,
                "message": (
                    "Signal is neutral; "
                    "no order was created"
                ),
            }

        # Register Layer 2 order with Layer 3
        order_manager.create_order(order)

        return {
            "status": "success",
            "order_created": True,
            "order": _order_response(order),
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# GET ORDER
# ============================================================

@router.get("/orders/{order_id}")
def get_order(
    order_id: str,
) -> dict:

    try:
        order = order_manager.get_order(order_id)

        return {
            "status": "success",
            "order": _order_response(order),
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


# ============================================================
# LIST ORDERS
# ============================================================

@router.get("/sessions/{session_id}/orders")
def list_session_orders(
    session_id: str,
) -> dict:

    try:
        engine.get_session(session_id)

        orders = order_manager.list_orders(
            session_id=session_id
        )

        return {
            "status": "success",
            "count": len(orders),
            "orders": [
                _order_response(order)
                for order in orders
            ],
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


# ============================================================
# CANCEL ORDER
# ============================================================

@router.post("/orders/{order_id}/cancel")
def cancel_order(
    order_id: str,
) -> dict:

    try:
        order = order_manager.cancel_order(order_id)

        return {
            "status": "success",
            "order": _order_response(order),
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# REJECT ORDER
# ============================================================

@router.post("/orders/{order_id}/reject")
def reject_order(
    order_id: str,
    reason: str = "Order rejected",
) -> dict:

    try:
        order = order_manager.reject_order(
            order_id,
            reason,
        )

        return {
            "status": "success",
            "order": _order_response(order),
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# EXECUTE ORDER
# ============================================================

@router.post("/orders/{order_id}/execute")
def execute_order(
    order_id: str,
    request: ExecuteOrderRequest,
) -> dict:

    try:
        order = order_manager.get_order(order_id)

        session_id = order.session_id

        if session_id not in execution_engines:
            raise KeyError(
                f"Execution engine not found for session {session_id}"
            )

        if session_id not in portfolios:
            raise KeyError(
                f"Portfolio not found for session {session_id}"
            )

        execution_engine = (
            execution_engines[session_id]
        )

        portfolio = portfolios[session_id]

        result = execution_engine.execute(
            order,
            market_price=request.market_price,
        )

        # Apply the filled order to the virtual portfolio
        position = portfolio.apply_order(order)

        return {
            "status": "success",
            "execution": {
                "order_id": result.order_id,
                "status": result.status.value,
                "requested_price": (
                    result.requested_price
                ),
                "execution_price": (
                    result.execution_price
                ),
                "requested_quantity": (
                    result.requested_quantity
                ),
                "executed_quantity": (
                    result.executed_quantity
                ),
                "gross_value": result.gross_value,
                "commission": result.commission,
                "slippage_cost": (
                    result.slippage_cost
                ),
                "total_cost": result.total_cost,
                "executed_at": result.executed_at,
            },
            "order": _order_response(order),
            "position": (
                _position_response(position)
                if position is not None
                else None
            ),
            "portfolio": (
                portfolio.get_summary()
            ),
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# GET PORTFOLIO
# ============================================================

@router.get("/sessions/{session_id}/portfolio")
def get_portfolio(
    session_id: str,
) -> dict:

    try:
        engine.get_session(session_id)

        if session_id not in portfolios:
            raise KeyError(
                f"Portfolio not found for session {session_id}"
            )

        portfolio = portfolios[session_id]

        return {
            "status": "success",
            "portfolio": portfolio.get_summary(),
            "positions": [
                _position_response(position)
                for position in portfolio.list_positions()
            ],
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


# ============================================================
# UPDATE MARKET PRICE
# ============================================================

@router.post(
    "/sessions/{session_id}/portfolio/price"
)
def update_market_price(
    session_id: str,
    request: UpdatePriceRequest,
) -> dict:

    try:
        if session_id not in portfolios:
            raise KeyError(
                f"Portfolio not found for session {session_id}"
            )

        portfolio = portfolios[session_id]

        positions = portfolio.list_positions()

        if not positions:
            raise ValueError(
                "No open positions available"
            )

        if len(positions) > 1:
            raise ValueError(
                "Multiple positions exist; use the "
                "instrument-specific portfolio update"
            )

        position = portfolio.update_market_price(
            positions[0].instrument_key,
            request.current_price,
        )

        return {
            "status": "success",
            "position": _position_response(position),
            "portfolio": portfolio.get_summary(),
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# METRICS
# ============================================================

@router.get("/sessions/{session_id}/metrics")
def get_metrics(
    session_id: str,
) -> dict:

    try:
        engine.get_session(session_id)

        if session_id not in portfolios:
            raise KeyError(
                f"Portfolio not found for session {session_id}"
            )

        portfolio = portfolios[session_id]

        trades = trade_histories[
            session_id
        ].list_trades()

        metrics = metrics_calculator.calculate(
            portfolio=portfolio,
            trades=trades,
        )

        return {
            "status": "success",
            "metrics": {
                "initial_capital": (
                    metrics.initial_capital
                ),
                "final_equity": (
                    metrics.final_equity
                ),
                "total_pnl": metrics.total_pnl,
                "realized_pnl": (
                    metrics.realized_pnl
                ),
                "unrealized_pnl": (
                    metrics.unrealized_pnl
                ),
                "total_return_percent": (
                    metrics.total_return_percent
                ),
                "total_trades": (
                    metrics.total_trades
                ),
                "winning_trades": (
                    metrics.winning_trades
                ),
                "losing_trades": (
                    metrics.losing_trades
                ),
                "breakeven_trades": (
                    metrics.breakeven_trades
                ),
                "win_rate_percent": (
                    metrics.win_rate_percent
                ),
                "average_win": (
                    metrics.average_win
                ),
                "average_loss": (
                    metrics.average_loss
                ),
                "profit_factor": (
                    metrics.profit_factor
                ),
                "maximum_drawdown": (
                    metrics.maximum_drawdown
                ),
                "maximum_drawdown_percent": (
                    metrics.maximum_drawdown_percent
                ),
            },
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


# ============================================================
# TRADE HISTORY
# ============================================================

@router.get("/sessions/{session_id}/trades")
def get_trade_history(
    session_id: str,
) -> dict:

    try:
        engine.get_session(session_id)

        if session_id not in trade_histories:
            raise KeyError(
                f"Trade history not found for session {session_id}"
            )

        trades = trade_histories[
            session_id
        ].list_trades()

        return {
            "status": "success",
            "count": len(trades),
            "trades": [
                {
                    "trade_id": trade.trade_id,
                    "session_id": trade.session_id,
                    "instrument_key": (
                        trade.instrument_key
                    ),
                    "side": trade.side.value,
                    "quantity": trade.quantity,
                    "entry_price": (
                        trade.entry_price
                    ),
                    "exit_price": (
                        trade.exit_price
                    ),
                    "entry_time": (
                        trade.entry_time
                    ),
                    "exit_time": (
                        trade.exit_time
                    ),
                    "gross_pnl": trade.gross_pnl,
                    "commission": trade.commission,
                    "slippage_cost": (
                        trade.slippage_cost
                    ),
                    "net_pnl": trade.net_pnl,
                    "signal": trade.signal,
                    "confidence": trade.confidence,
                }
                for trade in trades
            ],
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )