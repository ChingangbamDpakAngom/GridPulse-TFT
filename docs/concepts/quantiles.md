# Quantile Forecasting

## Why Quantiles?

Point forecasts (single number) hide uncertainty. For battery dispatch:

- **q10** (10th percentile): "Optimistic" — emissions likely *below* this → **safe to charge**
- **q50** (median): Most likely trajectory
- **q90** (90th percentile): "Pessimistic" — emissions likely *above* this → **avoid charging**

## Pinball Loss

For quantile $q \in (0,1)$, prediction $\hat{y}$, target $y$:

$$L_q(y, \hat{y}) = \begin{cases}
q (y - \hat{y}) & \text{if } y \ge \hat{y} \\
(q-1) (y - \hat{y}) & \text{if } y < \hat{y}
\end{cases}$$

Equivalently: $L_q = \max(q \cdot e, (q-1) \cdot e)$ where $e = y - \hat{y}$.

### Properties

- Asymmetric: under-prediction penalised differently from over-prediction
- Minimising expected pinball loss → consistent quantile estimator
- Differentiable almost everywhere → SGD compatible

## Multi-Horizon

Horizon $T=48$, quantiles $\mathcal{Q} = \{0.1, 0.5, 0.9\}$:

$$\mathcal{L} = \frac{1}{|\mathcal{Q}| T} \sum_{q \in \mathcal{Q}} \sum_{t=1}^{T} L_q(y_t, \hat{y}_t^{(q)})$$

Each horizon step contributes equally; no decay weighting.

## Calibration Check

After training, on held-out test set:

```python
def calibration_curve(y_true, y_pred_q10, y_pred_q50, y_pred_q90):
    for q, pred in [(0.1, y_pred_q10), (0.5, y_pred_q50), (0.9, y_pred_q90)]:
        coverage = (y_true <= pred).mean()
        print(f"q{q}: empirical coverage = {coverage:.3f} (target {q})")
```

Well-calibrated model → coverage ≈ quantile level.

## Decision Making with Quantiles

### Charge Window Selection (Post-Processor)

1. Compute historical median of q10 over recent windows
2. Slide 2h window across next 48h
3. Keep windows where `mean(q10) < historical_median`
4. Pick top 3 non-overlapping by lowest `mean(q10)`

This uses **only q10** — the optimistic bound — because we want *guaranteed* low-carbon periods.

### Risk-Aware Dispatch

| Strategy | Quantile Used | Use Case |
|----------|---------------|----------|
| Aggressive charging | q10 | Maximise renewable capture |
| Balanced | q50 | Expected cost minimisation |
| Conservative | q90 | Avoid high-carbon penalties |

## Related

- [[Model Architecture|Model Architecture]]
- [[Training|Training]]
- [[Post-Processing|Serving#post-processor]]