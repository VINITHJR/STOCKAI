# Phase 14 Advanced AI — Corrected Package

This package was audited against the uploaded Phase 14 implementation.

## Corrected files
- ai_decision_intelligence.py
- ai_confidence_reliability.py
- ai_recommendation_engine.py
- advanced_ai_validation.py
- explainable_ai_intelligence.py
- routes.py
- final_ai_intelligence_aggregator.py

Other Python/model files are retained from the uploaded package.

## Main corrections

1. Phase 9 AI signal + Phase 9 ensemble are correlated because both derive from
   the same RF/LSTM outputs. They are now combined inside one Layer 1 model
   evidence bucket instead of receiving separate top-level weights.

2. The current backend has no independent technical-analysis endpoint.
   Phase 14 no longer aliases the AI model signal as "technical" evidence.
   technical_signal is neutral until a real technical source is added.

3. Layer 2 reliability no longer includes Layer 1 decision confidence.
   This removes circular confidence -> reliability -> recommendation inflation.

4. Model reliability is based primarily on Phase 9 ensemble confidence and
   model agreement.

5. Supporting-analysis reliability no longer treats one duplicated technical
   signal as 100% confirmation.

6. Medium/high model conflict now affects recommendation confidence/strength
   and recommendation risk. Medium conflict cannot silently remain LOW risk.

7. When a directional Layer 1 result fails the directional threshold and is
   converted to HOLD, its confidence is the actual HOLD/neutral evidence score,
   not the old discarded directional score.

8. Layer 6 explicitly states that validation means internal pipeline
   consistency; it is not a prediction of future market performance.

9. Layer 7 metadata records the validation scope and the correlated model
   evidence architecture.

## Validation performed
- AST parse of all corrected Python files: passed.
- Python compileall of the corrected package inside the uploaded backend:
  passed.
- AdvancedAIOrchestrator integration test: passed.
- Phase 14 request-model import test: passed.

## Important
Sector direction remains neutral because the uploaded backend does not contain
a genuine sector-direction intelligence source. No synthetic sector direction
was introduced.
