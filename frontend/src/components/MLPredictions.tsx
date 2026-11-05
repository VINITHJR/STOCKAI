import { useState, useEffect } from "react";
import { PredictionCard, type LSTMPrediction } from "./PredictionCard";
import { TrendBadge, type TrendClassification } from "./TrendBadge";
import { CombinedInsight } from "./CombinedInsight";
import { AlertCircle } from "lucide-react";

interface MLPredictionsProps {
  instrumentKey: string;
  currentPrice: number;
}

interface CombinedPredictionResponse {
  status: string;
  instrument_key: string;
  price_prediction?: {
    status: string;
    current_price: number;
    predictions: LSTMPrediction[];
    model_type: string;
    timestamp: string;
  };
  trend_classification?: {
    status: string;
    current_price: number;
    trend: "bullish" | "bearish" | "neutral";
    confidence: number;
    probabilities: {
      bearish: number;
      neutral: number;
      bullish: number;
    };
    model_type: string;
    timestamp: string;
  };
  timestamp?: string;
}

export function MLPredictions({ instrumentKey, currentPrice }: MLPredictionsProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lstmPredictions, setLstmPredictions] = useState<LSTMPrediction[] | null>(null);
  const [trendClassification, setTrendClassification] = useState<TrendClassification | null>(null);

  useEffect(() => {
    let mounted = true;

    const fetchPredictions = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(`/api/combined/${instrumentKey}`);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data: CombinedPredictionResponse = await response.json();

        if (!mounted) return;

        if (data.status === "error") {
          throw new Error("Failed to fetch predictions");
        }

        // Extract LSTM predictions
        if (data.price_prediction && data.price_prediction.status === "success") {
          setLstmPredictions(data.price_prediction.predictions);
        } else {
          setLstmPredictions(null);
        }

        // Extract trend classification
        if (data.trend_classification && data.trend_classification.status === "success") {
          setTrendClassification({
            trend: data.trend_classification.trend,
            confidence: data.trend_classification.confidence,
            probabilities: data.trend_classification.probabilities,
          });
        } else {
          setTrendClassification(null);
        }

        setLoading(false);
      } catch (err) {
        if (!mounted) return;
        console.error("Error fetching predictions:", err);
        setError(err instanceof Error ? err.message : "Failed to load predictions");
        setLoading(false);
      }
    };

    fetchPredictions();

    // Refresh predictions every 5 minutes
    const interval = setInterval(fetchPredictions, 5 * 60 * 1000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [instrumentKey]);

  if (error) {
    return (
      <div className="ml-predictions-error">
        <AlertCircle size={24} />
        <div className="error-content">
          <h4>Predictions Unavailable</h4>
          <p>
            Models may not be trained yet. Train LSTM and Random Forest models to see predictions.
          </p>
          <button className="retry-button" onClick={() => window.location.reload()}>
            Retry
          </button>
        </div>

        <style>{`
          .ml-predictions-error {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 24px;
            background: var(--error-bg);
            border: 1px solid var(--error-border);
            border-radius: 12px;
            color: var(--error-color);
            margin-bottom: 16px;
          }

          .error-content h4 {
            margin: 0 0 8px 0;
            font-size: 16px;
            font-weight: 600;
          }

          .error-content p {
            margin: 0 0 12px 0;
            font-size: 14px;
            color: var(--text-secondary);
          }

          .retry-button {
            padding: 8px 16px;
            background: var(--error-color);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: opacity 0.2s;
          }

          .retry-button:hover {
            opacity: 0.9;
          }

          :root {
            --error-bg: #fef2f2;
            --error-border: #fecaca;
            --error-color: #ef4444;
            --text-secondary: #6b7280;
          }

          @media (prefers-color-scheme: dark) {
            :root:not([data-theme="light"]) {
              --error-bg: #450a0a;
              --error-border: #7f1d1d;
              --error-color: #ef4444;
              --text-secondary: #d1d5db;
            }
          }

          :root[data-theme="dark"] {
            --error-bg: #450a0a;
            --error-border: #7f1d1d;
            --error-color: #ef4444;
            --text-secondary: #d1d5db;
          }
        `}</style>
      </div>
    );
  }

  return (
    <div className="ml-predictions">
      <CombinedInsight
        lstmPredictions={lstmPredictions || undefined}
        trendClassification={trendClassification || undefined}
        currentPrice={currentPrice}
        loading={loading}
      />

      <div className="predictions-grid">
        <PredictionCard
          predictions={lstmPredictions || []}
          currentPrice={currentPrice}
          loading={loading}
        />

        <TrendBadge
          trend={trendClassification!}
          loading={loading}
        />
      </div>

      <style>{`
        .ml-predictions {
          width: 100%;
          margin-top: 24px;
        }

        .predictions-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 16px;
        }

        @media (min-width: 1024px) {
          .predictions-grid {
            grid-template-columns: 1fr 1fr;
          }
        }
      `}</style>
    </div>
  );
}
