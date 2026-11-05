import { TrendingUp, TrendingDown, Minus } from "lucide-react";

export interface LSTMPrediction {
  horizon_days: number;
  predicted_price: number;
  change_percent: number;
  confidence: number;
  lower_bound: number;
  upper_bound: number;
  direction: "bullish" | "bearish" | "neutral";
}

interface PredictionCardProps {
  predictions: LSTMPrediction[];
  currentPrice: number;
  loading?: boolean;
}

export function PredictionCard({
  predictions,
  currentPrice,
  loading = false,
}: PredictionCardProps) {
  if (loading) {
    return (
      <div className="prediction-card loading">
        <div className="prediction-header">
          <h3>📊 Price Predictions</h3>
          <span className="model-badge">LSTM Neural Network</span>
        </div>
        <div className="loading-skeleton">
          <div className="skeleton-line"></div>
          <div className="skeleton-line"></div>
          <div className="skeleton-line"></div>
        </div>
      </div>
    );
  }

  if (!predictions || predictions.length === 0) {
    return (
      <div className="prediction-card empty">
        <div className="prediction-header">
          <h3>📊 Price Predictions</h3>
          <span className="model-badge">LSTM Neural Network</span>
        </div>
        <p className="empty-message">No predictions available. Train the model first.</p>
      </div>
    );
  }

  const getDirectionIcon = (direction: string) => {
    switch (direction) {
      case "bullish":
        return <TrendingUp className="icon-bullish" size={16} />;
      case "bearish":
        return <TrendingDown className="icon-bearish" size={16} />;
      default:
        return <Minus className="icon-neutral" size={16} />;
    }
  };

  const getDirectionColor = (direction: string) => {
    switch (direction) {
      case "bullish":
        return "bullish";
      case "bearish":
        return "bearish";
      default:
        return "neutral";
    }
  };

  return (
    <div className="prediction-card">
      <div className="prediction-header">
        <h3>📊 Price Predictions</h3>
        <span className="model-badge">LSTM Neural Network</span>
      </div>

      <div className="current-price-section">
        <span className="label">Current Price</span>
        <span className="current-price">₹{currentPrice.toFixed(2)}</span>
      </div>

      <div className="predictions-list">
        {predictions.map((pred) => (
          <div key={pred.horizon_days} className="prediction-item">
            <div className="prediction-info">
              <div className="prediction-header-row">
                <span className="horizon">{pred.horizon_days}d Target</span>
                {getDirectionIcon(pred.direction)}
              </div>

              <div className="prediction-values">
                <span className={`predicted-price ${getDirectionColor(pred.direction)}`}>
                  ₹{pred.predicted_price.toFixed(2)}
                </span>
                <span className={`change-percent ${getDirectionColor(pred.direction)}`}>
                  {pred.change_percent >= 0 ? "+" : ""}
                  {pred.change_percent.toFixed(2)}%
                </span>
              </div>

              <div className="prediction-range">
                <span className="range-label">Range:</span>
                <span className="range-values">
                  ₹{pred.lower_bound.toFixed(2)} - ₹{pred.upper_bound.toFixed(2)}
                </span>
              </div>
            </div>

            <div className="confidence-section">
              <div className="confidence-label">
                <span>Confidence</span>
                <span className="confidence-value">{(pred.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{ width: `${pred.confidence * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <style>{`
        .prediction-card {
          background: var(--card-bg);
          border: 1px solid var(--border-color);
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 16px;
        }

        .prediction-card.loading,
        .prediction-card.empty {
          min-height: 200px;
          display: flex;
          flex-direction: column;
          justify-content: center;
        }

        .prediction-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }

        .prediction-header h3 {
          font-size: 18px;
          font-weight: 600;
          color: var(--text-primary);
          margin: 0;
        }

        .model-badge {
          background: var(--accent-bg);
          color: var(--accent-text);
          padding: 4px 12px;
          border-radius: 6px;
          font-size: 11px;
          font-weight: 500;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .current-price-section {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px;
          background: var(--subtle-bg);
          border-radius: 8px;
          margin-bottom: 16px;
        }

        .current-price-section .label {
          font-size: 13px;
          color: var(--text-secondary);
          font-weight: 500;
        }

        .current-price {
          font-size: 20px;
          font-weight: 700;
          color: var(--text-primary);
        }

        .predictions-list {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .prediction-item {
          padding: 16px;
          background: var(--subtle-bg);
          border-radius: 10px;
          border: 1px solid var(--border-light);
        }

        .prediction-info {
          margin-bottom: 12px;
        }

        .prediction-header-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }

        .horizon {
          font-size: 13px;
          font-weight: 600;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .icon-bullish {
          color: var(--success-color);
        }

        .icon-bearish {
          color: var(--danger-color);
        }

        .icon-neutral {
          color: var(--text-tertiary);
        }

        .prediction-values {
          display: flex;
          align-items: baseline;
          gap: 12px;
          margin-bottom: 6px;
        }

        .predicted-price {
          font-size: 24px;
          font-weight: 700;
        }

        .predicted-price.bullish {
          color: var(--success-color);
        }

        .predicted-price.bearish {
          color: var(--danger-color);
        }

        .predicted-price.neutral {
          color: var(--text-primary);
        }

        .change-percent {
          font-size: 16px;
          font-weight: 600;
        }

        .change-percent.bullish {
          color: var(--success-color);
        }

        .change-percent.bearish {
          color: var(--danger-color);
        }

        .change-percent.neutral {
          color: var(--text-secondary);
        }

        .prediction-range {
          display: flex;
          gap: 6px;
          font-size: 12px;
          color: var(--text-tertiary);
        }

        .range-label {
          font-weight: 500;
        }

        .confidence-section {
          margin-top: 12px;
        }

        .confidence-label {
          display: flex;
          justify-content: space-between;
          font-size: 12px;
          color: var(--text-secondary);
          margin-bottom: 6px;
        }

        .confidence-value {
          font-weight: 600;
          color: var(--text-primary);
        }

        .confidence-bar {
          width: 100%;
          height: 6px;
          background: var(--bar-bg);
          border-radius: 3px;
          overflow: hidden;
        }

        .confidence-fill {
          height: 100%;
          background: linear-gradient(90deg, var(--accent-color), var(--accent-light));
          border-radius: 3px;
          transition: width 0.3s ease;
        }

        .loading-skeleton {
          display: flex;
          flex-direction: column;
          gap: 16px;
          padding: 20px 0;
        }

        .skeleton-line {
          height: 60px;
          background: linear-gradient(
            90deg,
            var(--skeleton-base) 25%,
            var(--skeleton-shimmer) 50%,
            var(--skeleton-base) 75%
          );
          background-size: 200% 100%;
          animation: shimmer 1.5s infinite;
          border-radius: 8px;
        }

        @keyframes shimmer {
          0% {
            background-position: 200% 0;
          }
          100% {
            background-position: -200% 0;
          }
        }

        .empty-message {
          text-align: center;
          color: var(--text-secondary);
          font-size: 14px;
          padding: 40px 20px;
        }

        /* Dark/Light theme variables */
        :root {
          --card-bg: #ffffff;
          --border-color: #e5e7eb;
          --border-light: #f3f4f6;
          --text-primary: #111827;
          --text-secondary: #6b7280;
          --text-tertiary: #9ca3af;
          --subtle-bg: #f9fafb;
          --accent-bg: #dbeafe;
          --accent-text: #1e40af;
          --accent-color: #3b82f6;
          --accent-light: #60a5fa;
          --success-color: #10b981;
          --danger-color: #ef4444;
          --bar-bg: #e5e7eb;
          --skeleton-base: #f3f4f6;
          --skeleton-shimmer: #e5e7eb;
        }

        @media (prefers-color-scheme: dark) {
          :root:not([data-theme="light"]) {
            --card-bg: #1f2937;
            --border-color: #374151;
            --border-light: #374151;
            --text-primary: #f9fafb;
            --text-secondary: #d1d5db;
            --text-tertiary: #9ca3af;
            --subtle-bg: #111827;
            --accent-bg: #1e3a8a;
            --accent-text: #93c5fd;
            --accent-color: #3b82f6;
            --accent-light: #60a5fa;
            --success-color: #10b981;
            --danger-color: #ef4444;
            --bar-bg: #374151;
            --skeleton-base: #374151;
            --skeleton-shimmer: #4b5563;
          }
        }

        :root[data-theme="dark"] {
          --card-bg: #1f2937;
          --border-color: #374151;
          --border-light: #374151;
          --text-primary: #f9fafb;
          --text-secondary: #d1d5db;
          --text-tertiary: #9ca3af;
          --subtle-bg: #111827;
          --accent-bg: #1e3a8a;
          --accent-text: #93c5fd;
          --accent-color: #3b82f6;
          --accent-light: #60a5fa;
          --success-color: #10b981;
          --danger-color: #ef4444;
          --bar-bg: #374151;
          --skeleton-base: #374151;
          --skeleton-shimmer: #4b5563;
        }
      `}</style>
    </div>
  );
}
