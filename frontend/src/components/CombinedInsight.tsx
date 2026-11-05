import { Brain, Activity, AlertCircle, CheckCircle, Info } from "lucide-react";
import type { LSTMPrediction } from "./PredictionCard";
import type { TrendClassification } from "./TrendBadge";

interface CombinedInsightProps {
  lstmPredictions?: LSTMPrediction[];
  trendClassification?: TrendClassification;
  currentPrice: number;
  loading?: boolean;
}

export function CombinedInsight({
  lstmPredictions,
  trendClassification,
  loading = false,
}: CombinedInsightProps) {
  if (loading) {
    return (
      <div className="combined-insight loading">
        <div className="insight-header">
          <Brain size={20} />
          <h3>AI Market Outlook</h3>
        </div>
        <div className="loading-skeleton">
          <div className="skeleton-line"></div>
          <div className="skeleton-line short"></div>
        </div>
      </div>
    );
  }

  if (!lstmPredictions && !trendClassification) {
    return (
      <div className="combined-insight empty">
        <div className="insight-header">
          <Brain size={20} />
          <h3>AI Market Outlook</h3>
        </div>
        <p className="empty-message">Train both models to see combined insights.</p>
      </div>
    );
  }

  // Generate insight based on both models
  const generateInsight = () => {
    if (!lstmPredictions || !trendClassification) {
      return {
        signal: "incomplete",
        strength: "unknown",
        message: "Incomplete prediction data",
        details: "Train both models for complete analysis.",
        icon: <Info size={20} />,
      };
    }

    const sevenDayPred = lstmPredictions.find((p) => p.horizon_days === 7);
    if (!sevenDayPred) {
      return {
        signal: "incomplete",
        strength: "unknown",
        message: "Insufficient prediction horizon",
        details: "7-day forecast not available.",
        icon: <Info size={20} />,
      };
    }

    const priceChange = sevenDayPred.change_percent;
    const priceDirection = sevenDayPred.direction;
    const trendDirection = trendClassification.trend;
    const trendConfidence = trendClassification.confidence;

    // Both models agree on bullish
    if (priceDirection === "bullish" && trendDirection === "bullish") {
      const strength = trendConfidence >= 0.7 ? "strong" : "moderate";
      return {
        signal: "bullish",
        strength,
        message: strength === "strong"
          ? "Strong Bullish Signal"
          : "Moderate Bullish Signal",
        details: `Both LSTM and Random Forest models agree on upward movement. Expected gain of ${priceChange.toFixed(2)}% over 7 days with ${(trendConfidence * 100).toFixed(0)}% trend confidence.`,
        icon: <CheckCircle size={20} />,
      };
    }

    // Both models agree on bearish
    if (priceDirection === "bearish" && trendDirection === "bearish") {
      const strength = trendConfidence >= 0.7 ? "strong" : "moderate";
      return {
        signal: "bearish",
        strength,
        message: strength === "strong"
          ? "Strong Bearish Signal"
          : "Moderate Bearish Signal",
        details: `Both models indicate downward pressure. Expected decline of ${Math.abs(priceChange).toFixed(2)}% over 7 days with ${(trendConfidence * 100).toFixed(0)}% trend confidence.`,
        icon: <CheckCircle size={20} />,
      };
    }

    // Price bullish, trend neutral/bearish
    if (priceDirection === "bullish" && trendDirection !== "bullish") {
      return {
        signal: "mixed",
        strength: "conflicting",
        message: "Mixed Signals Detected",
        details: `LSTM predicts price rise (${priceChange.toFixed(2)}%) but trend momentum is ${trendDirection}. Exercise caution and wait for confirmation.`,
        icon: <AlertCircle size={20} />,
      };
    }

    // Price bearish, trend neutral/bullish
    if (priceDirection === "bearish" && trendDirection !== "bearish") {
      return {
        signal: "mixed",
        strength: "conflicting",
        message: "Mixed Signals Detected",
        details: `LSTM predicts price decline (${priceChange.toFixed(2)}%) but trend shows ${trendDirection} characteristics. Monitor closely for reversal.`,
        icon: <AlertCircle size={20} />,
      };
    }

    // Neutral scenarios
    if (trendDirection === "neutral") {
      const absChange = Math.abs(priceChange);
      if (absChange < 2) {
        return {
          signal: "neutral",
          strength: "weak",
          message: "Consolidation Phase",
          details: `Low volatility expected. Price movement minimal (${priceChange.toFixed(2)}%) with neutral trend. Consider waiting for clearer signals.`,
          icon: <Activity size={20} />,
        };
      } else {
        return {
          signal: "mixed",
          strength: "uncertain",
          message: "Uncertain Outlook",
          details: `LSTM shows ${absChange.toFixed(2)}% movement but trend is neutral. Market direction unclear, trade with caution.`,
          icon: <AlertCircle size={20} />,
        };
      }
    }

    return {
      signal: "neutral",
      strength: "weak",
      message: "Neutral Market Outlook",
      details: "No strong directional bias detected. Monitor for breakout signals.",
      icon: <Activity size={20} />,
    };
  };

  const insight = generateInsight();

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case "bullish":
        return "bullish";
      case "bearish":
        return "bearish";
      case "mixed":
        return "mixed";
      default:
        return "neutral";
    }
  };

  const getRecommendation = () => {
    if (!lstmPredictions || !trendClassification) return null;

    switch (insight.signal) {
      case "bullish":
        return insight.strength === "strong"
          ? "Consider long positions with proper risk management."
          : "Cautiously bullish. Wait for additional confirmation.";
      case "bearish":
        return insight.strength === "strong"
          ? "Consider reducing exposure or defensive positions."
          : "Cautiously bearish. Monitor support levels closely.";
      case "mixed":
        return "Conflicting signals suggest waiting for clearer direction.";
      default:
        return "Sideways movement expected. Consider range-bound strategies.";
    }
  };

  return (
    <div className={`combined-insight ${getSignalColor(insight.signal)}`}>
      <div className="insight-header">
        <Brain size={20} />
        <h3>AI Market Outlook</h3>
      </div>

      <div className="insight-main">
        <div className="signal-badge">
          {insight.icon}
          <span className="signal-message">{insight.message}</span>
        </div>

        <p className="insight-details">{insight.details}</p>

        <div className="recommendation-section">
          <div className="recommendation-label">
            <Activity size={14} />
            <span>Trading Recommendation</span>
          </div>
          <p className="recommendation-text">{getRecommendation()}</p>
        </div>
      </div>

      <div className="model-agreement">
        <div className="agreement-item">
          <span className="label">Price Forecast</span>
          <span className={`value ${lstmPredictions?.[0]?.direction}`}>
            {lstmPredictions?.[0]?.direction?.toUpperCase() || "N/A"}
          </span>
        </div>
        <div className="agreement-divider">×</div>
        <div className="agreement-item">
          <span className="label">Trend Signal</span>
          <span className={`value ${trendClassification?.trend}`}>
            {trendClassification?.trend?.toUpperCase() || "N/A"}
          </span>
        </div>
      </div>

      <style>{`
        .combined-insight {
          background: var(--card-bg);
          border: 1px solid var(--border-color);
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 16px;
        }

        .combined-insight.loading,
        .combined-insight.empty {
          min-height: 150px;
          display: flex;
          flex-direction: column;
          justify-content: center;
        }

        .insight-header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 16px;
          color: var(--text-primary);
        }

        .insight-header h3 {
          font-size: 18px;
          font-weight: 600;
          margin: 0;
        }

        .insight-main {
          padding: 16px;
          background: var(--subtle-bg);
          border-radius: 10px;
          margin-bottom: 16px;
          border-left: 4px solid var(--accent-color);
        }

        .combined-insight.bullish .insight-main {
          border-left-color: var(--success-color);
          background: var(--success-subtle-bg);
        }

        .combined-insight.bearish .insight-main {
          border-left-color: var(--danger-color);
          background: var(--danger-subtle-bg);
        }

        .combined-insight.mixed .insight-main {
          border-left-color: var(--warning-color);
          background: var(--warning-subtle-bg);
        }

        .signal-badge {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 12px;
          padding: 12px;
          background: var(--card-bg);
          border-radius: 8px;
        }

        .combined-insight.bullish .signal-badge {
          color: var(--success-color);
        }

        .combined-insight.bearish .signal-badge {
          color: var(--danger-color);
        }

        .combined-insight.mixed .signal-badge {
          color: var(--warning-color);
        }

        .combined-insight.neutral .signal-badge {
          color: var(--text-secondary);
        }

        .signal-message {
          font-size: 16px;
          font-weight: 700;
        }

        .insight-details {
          font-size: 14px;
          line-height: 1.6;
          color: var(--text-primary);
          margin: 0 0 16px 0;
        }

        .recommendation-section {
          padding: 12px;
          background: var(--card-bg);
          border-radius: 8px;
        }

        .recommendation-label {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 12px;
          font-weight: 600;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-bottom: 8px;
        }

        .recommendation-text {
          font-size: 14px;
          font-weight: 500;
          color: var(--text-primary);
          margin: 0;
        }

        .model-agreement {
          display: flex;
          align-items: center;
          justify-content: space-around;
          padding: 12px;
          background: var(--subtle-bg);
          border-radius: 8px;
        }

        .agreement-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 6px;
        }

        .agreement-item .label {
          font-size: 11px;
          font-weight: 500;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .agreement-item .value {
          font-size: 14px;
          font-weight: 700;
          padding: 4px 12px;
          border-radius: 6px;
        }

        .agreement-item .value.bullish {
          color: var(--success-color);
          background: var(--success-bg);
        }

        .agreement-item .value.bearish {
          color: var(--danger-color);
          background: var(--danger-bg);
        }

        .agreement-item .value.neutral {
          color: var(--neutral-color);
          background: var(--neutral-bg);
        }

        .agreement-divider {
          font-size: 20px;
          font-weight: 300;
          color: var(--text-tertiary);
        }

        .loading-skeleton {
          display: flex;
          flex-direction: column;
          gap: 12px;
          padding: 20px 0;
        }

        .skeleton-line {
          height: 20px;
          background: linear-gradient(
            90deg,
            var(--skeleton-base) 25%,
            var(--skeleton-shimmer) 50%,
            var(--skeleton-base) 75%
          );
          background-size: 200% 100%;
          animation: shimmer 1.5s infinite;
          border-radius: 4px;
        }

        .skeleton-line.short {
          width: 70%;
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
          padding: 30px 20px;
        }

        /* Dark/Light theme variables */
        :root {
          --card-bg: #ffffff;
          --border-color: #e5e7eb;
          --text-primary: #111827;
          --text-secondary: #6b7280;
          --text-tertiary: #9ca3af;
          --subtle-bg: #f9fafb;
          --accent-color: #3b82f6;
          --success-color: #10b981;
          --success-bg: #d1fae5;
          --success-subtle-bg: #f0fdf4;
          --danger-color: #ef4444;
          --danger-bg: #fee2e2;
          --danger-subtle-bg: #fef2f2;
          --warning-color: #f59e0b;
          --warning-subtle-bg: #fffbeb;
          --neutral-color: #6b7280;
          --neutral-bg: #f3f4f6;
          --skeleton-base: #f3f4f6;
          --skeleton-shimmer: #e5e7eb;
        }

        @media (prefers-color-scheme: dark) {
          :root:not([data-theme="light"]) {
            --card-bg: #1f2937;
            --border-color: #374151;
            --text-primary: #f9fafb;
            --text-secondary: #d1d5db;
            --text-tertiary: #9ca3af;
            --subtle-bg: #111827;
            --accent-color: #3b82f6;
            --success-color: #10b981;
            --success-bg: #064e3b;
            --success-subtle-bg: #022c22;
            --danger-color: #ef4444;
            --danger-bg: #7f1d1d;
            --danger-subtle-bg: #450a0a;
            --warning-color: #f59e0b;
            --warning-subtle-bg: #78350f;
            --neutral-color: #9ca3af;
            --neutral-bg: #374151;
            --skeleton-base: #374151;
            --skeleton-shimmer: #4b5563;
          }
        }

        :root[data-theme="dark"] {
          --card-bg: #1f2937;
          --border-color: #374151;
          --text-primary: #f9fafb;
          --text-secondary: #d1d5db;
          --text-tertiary: #9ca3af;
          --subtle-bg: #111827;
          --accent-color: #3b82f6;
          --success-color: #10b981;
          --success-bg: #064e3b;
          --success-subtle-bg: #022c22;
          --danger-color: #ef4444;
          --danger-bg: #7f1d1d;
          --danger-subtle-bg: #450a0a;
          --warning-color: #f59e0b;
          --warning-subtle-bg: #78350f;
          --neutral-color: #9ca3af;
          --neutral-bg: #374151;
          --skeleton-base: #374151;
          --skeleton-shimmer: #4b5563;
        }
      `}</style>
    </div>
  );
}
