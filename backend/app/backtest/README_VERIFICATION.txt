Phase 10 Verified Layer 1, Layer 3 Part 3, Layer 4 Part 1/3, Layer 5 Part 1/2/3
====================================================================================

Files:
- ai_layer1.py
- ai_layer3_metrics.py
- ai_layer4.py
- ai_layer4_metrics.py
- ai_layer5.py
- ai_layer5_compare.py
- ai_layer5_metrics.py

Verified changes:
1. Layer 3 drawdown now exposes a true percentage metric (`simulated_max_drawdown` / `average_max_drawdown`)
   and also preserves the absolute amount as `*_amount`. This fixes the dashboard showing an absolute
   capital drawdown as a percentage.
2. Layer 4 summary now actually exposes positive/negative return counts and average forward return.
   The previous Part 3 adapter looked for these fields in `summary`, but Part 1 only placed the counts
   inside grouped summaries, which caused the dashboard to show 0.
3. Layer 1 now emits explicit model_agreement/model_conflict, available_candle_count and
   future_data_used_for_signal fields, making downstream validation auditable.
4. Layer 1 drawdown also exposes both percent and absolute amount.
5. Layer 3 boolean fields are handled strictly as booleans instead of treating strings such as "false"
   as truthy.
6. Layer 5 Part 2 now validates that Part 1 observations match Layer 3 records by period/timestamp,
   preventing silent misalignment.
7. Layer 5 uses the same threshold-derived actual direction consistently rather than trusting a possibly
   stale copied `actual_direction`.
8. Numeric validation rejects non-finite configuration values.

Verification performed:
- Python compilation of all 7 files: PASS
- Synthetic Layer 3 -> Layer 4 -> Layer 5 integration: PASS
- Layer 5 record mismatch guard: PASS
- Drawdown invariant stress test (100 random simulations): PASS

Important:
- Layer 3 Part 2 was NOT supplied in the uploaded files, so it was not modified.
- Layer 6/frontend was NOT modified in this package.
- Back up your current files before replacing them.
