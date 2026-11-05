import { TrendingUp, TrendingDown, Minus, Target } from "lucide-react";

export interface TrendClassification {
  trend: "bullish" | "bearish" | "neutral";
  confidence: number;
  probabilities: {
    bearish: number;
    neutral: number;
    bullish: number;
  };
}

interface TrendBadgeProps {
  trend: TrendClassification;
  loading?: boolean;
}

export function TrendBadge({ trend, loading = false }: TrendBadgeProps) {
  if (loading) {
    return (
      <div className="trend-badge loading">
        <div className="trend-header">
          <h3>🎯 Trend Classification</h3>
          <span className="model-badge">Random Forest</span>
        </div>
        <div className="loading-skeleton">
          <div className="skeleton-circle"></div>
          <div className="skeleton-line"></div>
        </div>
      </div>
    );
  }

  if (!trend) {
    return (
      <div className="trend-badge empty">
        <div className="trend-header">
          <h3>🎯 Trend Classification</h3>
          <span className="model-badge">Random Forest</span>
        </div>
        <p className="empty-message">No trend data available. Train the model first.</p>
      </div>
    );
  }

  const getTrendIcon = (trendType: string) => {
    switch (trendType) {
      case "bullish":
        return <TrendingUp size={32} />;
      case "bearish":
        return <TrendingDown size={32} />;
      default:
        return <Minus size={32} />;
    }
  };

  const getTrendColor = (trendType: string) => {
    switch (trendType) {
      case "bullish":
        return "bullish";
      case "bearish":
        return "bearish";
      default:
        return "neutral";
    }
  };

  const getTrendLabel = (trendType: string) => {
    switch (trendType) {
      case "bullish":
        return "Bullish";
      case "bearish":
        return "Bearish";
      default:
        return "Neutral";
    }
  };

  const getConfidenceLevel = (confidence: number) => {
    if (confidence >= 0.8) return "Very High";
    if (confidence >= 0.7) return "High";
    if (confidence >= 0.6) return "Moderate";
    if (confidence >= 0.5) return "Low";
    return "Very Low";
  };

  const trendColor = getTrendColor(trend.trend);

  return (
    <div className="trend-badge">
      <div className="trend-header">
        <h3>🎯 Trend Classification</h3>
        <span className="model-badge">Random Forest</span>
      </div>

      <div className={`trend-main ${trendColor}`}>
        <div className="trend-icon-wrapper">
          {getTrendIcon(trend.trend)}
        </div>
        <div className="trend-details">
          <div className="trend-label">{getTrendLabel(trend.trend)}</div>
          <div className="confidence-main">
            <span className="confidence-value">{(trend.confidence * 100).toFixed(0)}%</span>
            <span className="confidence-level">Confidence</span>
          </div>
          <div className="confidence-level-badge">
            {getConfidenceLevel(trend.confidence)}
          </div>
        </div>
      </div>

      <div className="probabilities-section">
        <div className="probabilities-header">
          <Target size={14} />
          <span>Probability Breakdown</span>
        </div>

        <div className="probability-bars">
          <div className="probability-item">
            <div className="probability-label">
              <span className="label-text bearish-text">Bearish</span>
              <span className="label-percent">{(trend.probabilities.bearish * 100).toFixed(0)}%</span>
            </div>
            <div className="probability-bar">
              <div
                className="probability-fill bearish-fill"
                style={{ width: `${trend.probabilities.bearish * 100}%` }}
              ></div>
            </div>
          </div>

          <div className="probability-item">
            <div className="probability-label">
              <span className="label-text neutral-text">Neutral</span>
              <span className="label-percent">{(trend.probabilities.neutral * 100).toFixed(0)}%</span>
            </div>
            <div className="probability-bar">
              <div
                className="probability-fill neutral-fill"
                style={{ width: `${trend.probabilities.neutral * 100}%` }}
              ></div>
            </div>
          </div>

          <div className="probability-item">
            <div className="probability-label">
              <span className="label-text bullish-text">Bullish</span>
              <span className="label-percent">{(trend.probabilities.bullish * 100).toFixed(0)}%</span>
            </div>
            <div className="probability-bar">
              <div
                className="probability-fill bullish-fill"
                style={{ width: `${trend.probabilities.bullish * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .trend-badge {
          background: var(--card-bg);
          border: 1px solid var(--border-color);
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 16px;
        }

        .trend-badge.loading,
        .trend-badge.empty {
          min-height: 200px;
          display: flex;
          flex-direction: column;
          justify-content: center;
        }

        .trend-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }

        .trend-header h3 {
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

        .trend-main {
          display: flex;
          align-items: center;
          gap: 20px;
          padding: 20px;
          border-radius: 10px;
          margin-bottom: 20px;
          border: 2px solid;
          transition: all 0.3s ease;
        }

        .trend-main.bullish {
          background: var(--success-bg);
          border-color: var(--success-color);
        }

        .trend-main.bearish {
          background: var(--danger-bg);
          border-color: var(--danger-color);
        }

        .trend-main.neutral {
          background: var(--neutral-bg);
          border-color: var(--neutral-color);
        }

        .trend-icon-wrapper {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 64px;
          height: 64px;
          border-radius: 50%;
          background: var(--card-bg);
          flex-shrink: 0;
        }

        .trend-main.bullish .trend-icon-wrapper {
          color: var(--success-color);
        }

        .trend-main.bearish .trend-icon-wrapper {
          color: var(--danger-color);
        }

        .trend-main.neutral .trend-icon-wrapper {
          color: var(--neutral-color);
        }

        .trend-details {
          flex: 1;
        }

        .trend-label {
          font-size: 24px;
          font-weight: 700;
          margin-bottom: 8px;
        }

        .trend-main.bullish .trend-label {
          color: var(--success-color);
        }

        .trend-main.bearish .trend-label {
          color: var(--danger-color);
        }

        .trend-main.neutral .trend-label {
          color: var(--neutral-color);
        }

        .confidence-main {
          display: flex;
          align-items: baseline;
          gap: 8px;
          margin-bottom: 8px;
        }

        .confidence-value {
          font-size: 32px;
          font-weight: 700;
          color: var(--text-primary);
        }

        .confidence-level {
          font-size: 14px;
          color: var(--text-secondary);
          font-weight: 500;
        }

        .confidence-level-badge {
          display: inline-block;
          padding: 4px 12px;
          border-radius: 6px;
          font-size: 12px;
          font-weight: 600;
          background: var(--card-bg);
          color: var(--text-primary);
        }

        .probabilities-section {
          padding: 16px;
          background: var(--subtle-bg);
          border-radius: 10px;
        }

        .probabilities-header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 16px;
          font-size: 13px;
          font-weight: 600;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .probability-bars {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .probability-item {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .probability-label {
          display: flex;
          justify-content: space-between;
          font-size: 13px;
          font-weight: 500;
        }

        .label-text {
          font-weight: 600;
        }

        .bearish-text {
          color: var(--danger-color);
        }

        .neutral-text {
          color: var(--neutral-color);
        }

        .bullish-text {
          color: var(--success-color);
        }

        .label-percent {
          color: var(--text-primary);
          font-weight: 600;
        }

        .probability-bar {
          width: 100%;
          height: 8px;
          background: var(--bar-bg);
          border-radius: 4px;
          overflow: hidden;
        }

        .probability-fill {
          height: 100%;
          border-radius: 4px;
          transition: width 0.5s ease;
        }

        .bearish-fill {
          background: linear-gradient(90deg, var(--danger-color), var(--danger-light));
        }

        .neutral-fill {
          background: linear-gradient(90deg, var(--neutral-color), var(--neutral-light));
        }

        .bullish-fill {
          background: linear-gradient(90deg, var(--success-color), var(--success-light));
        }

        .loading-skeleton {
          display: flex;
          align-items: center;
          gap: 20px;
          padding: 20px;
        }

        .skeleton-circle {
          width: 64px;
          height: 64px;
          border-radius: 50%;
          background: linear-gradient(
            90deg,
            var(--skeleton-base) 25%,
            var(--skeleton-shimmer) 50%,
            var(--skeleton-base) 75%
          );
          background-size: 200% 100%;
          animation: shimmer 1.5s infinite;
        }

        .skeleton-line {
          flex: 1;
          height: 40px;
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
          --text-primary: #111827;
          --text-secondary: #6b7280;
          --subtle-bg: #f9fafb;
          --accent-bg: #dbeafe;
          --accent-text: #1e40af;
          --success-color: #10b981;
          --success-light: #34d399;
          --success-bg: #d1fae5;
          --danger-color: #ef4444;
          --danger-light: #f87171;
          --danger-bg: #fee2e2;
          --neutral-color: #f59e0b;
          --neutral-light: #fbbf24;
          --neutral-bg: #fef3c7;
          --bar-bg: #e5e7eb;
          --skeleton-base: #f3f4f6;
          --skeleton-shimmer: #e5e7eb;
        }

        @media (prefers-color-scheme: dark) {
          :root:not([data-theme="light"]) {
            --card-bg: #1f2937;
            --border-color: #374151;
            --text-primary: #f9fafb;
            --text-secondary: #d1d5db;
            --subtle-bg: #111827;
            --accent-bg: #1e3a8a;
            --accent-text: #93c5fd;
            --success-color: #10b981;
            --success-light: #34d399;
            --success-bg: #064e3b;
            --danger-color: #ef4444;
            --danger-light: #f87171;
            --danger-bg: #7f1d1d;
            --neutral-color: #f59e0b;
            --neutral-light: #fbbf24;
            --neutral-bg: #78350f;
            --bar-bg: #374151;
            --skeleton-base: #374151;
            --skeleton-shimmer: #4b5563;
          }
        }

        :root[data-theme="dark"] {
          --card-bg: #1f2937;
          --border-color: #374151;
          --text-primary: #f9fafb;
          --text-secondary: #d1d5db;
          --subtle-bg: #111827;
          --accent-bg: #1e3a8a;
          --accent-text: #93c5fd;
          --success-color: #10b981;
          --success-light: #34d399;
          --success-bg: #064e3b;
          --danger-color: #ef4444;
          --danger-light: #f87171;
          --danger-bg: #7f1d1d;
          --neutral-color: #f59e0b;
          --neutral-light: #fbbf24;
          --neutral-bg: #78350f;
          --bar-bg: #374151;
          --skeleton-base: #374151;
          --skeleton-shimmer: #4b5563;
        }
      `}</style>
    </div>
  );
}
