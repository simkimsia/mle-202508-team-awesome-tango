# XGBoost vs LSTM: Sample Size Considerations

This document explains why XGBoost and LSTM have fundamentally different statistical requirements when working with the same N-CMAPSS dataset splitting strategy described in `01-data-splitting-strategy.md`.

## The Core Question

Given that the LSTM model is limited by the small number of engines in validation (7), test (9), and OOT (19) sets, does XGBoost face the same limitation when using the same data splitting strategy?

**Answer: No, XGBoost does NOT face the same limitation.**

## Why LSTM is Limited by Number of Engines

### How LSTM Processes Data

LSTM (Long Short-Term Memory) is a recurrent neural network that:

- Processes data **sequentially** as time series
- Treats each engine as a **complete trajectory/sequence**
- Learns **temporal dependencies** across the entire lifecycle of each engine
- Models the evolution of sensor readings over time within each engine

### Statistical Unit for LSTM

- **Statistical unit** = One complete engine sequence
- The model learns patterns across the temporal dimension of each engine's lifecycle
- Each engine represents one independent time series for evaluation

### LSTM's Sample Size Problem

```
Training Set:    39 engines = 39 independent sequences   ✓ (n ≥ 30)
Validation Set:   7 engines =  7 independent sequences   ✗ (n < 30)
Test Set:         9 engines =  9 independent sequences   ✗ (n < 30)
OOT Set:         19 engines = 19 independent sequences   ✗ (n < 30)
```

With only 7 validation engines, LSTM has only **7 independent sequences** for model selection and hyperparameter tuning, which is statistically insufficient according to the Central Limit Theorem (n ≥ 30).

## Why XGBoost is NOT Limited by Number of Engines

### How XGBoost Processes Data

XGBoost (Extreme Gradient Boosting) is a tree-based ensemble method that:

- Treats each row as an **observation** (though not fully independent due to rolling windows)
- Does NOT inherently model temporal/sequential dependencies like LSTM
- Learns patterns from **time-windowed features** at each time step
- Makes predictions based on engineered features that capture temporal patterns

### Statistical Unit for XGBoost

**IMPORTANT: The N-CMAPSS XGBoost implementation uses rolling window features:**

- Rolling mean and std for windows of **5, 15, and 30 time steps**
- This creates **temporal dependencies** between consecutive rows
- Row at time `t` shares data with rows at times `t-1` through `t-29` (for 30-window)

This means observations are **NOT fully independent**, but they are much less dependent than LSTM sequences.

### XGBoost's Sample Size: The Full Picture

**Naive count (treating all rows as independent):**
```
Training Set:    39 engines = 26,898,226 rows
Validation Set:   7 engines =  6,905,795 rows
Test Set:         9 engines =  7,333,005 rows
OOT Set:         19 engines = 13,737,152 rows
```

**Effective independent sample size (accounting for 30-step rolling window):**

Since the longest window is 30 steps, consecutive observations share 29 overlapping time steps. To estimate truly independent samples, we divide by the window size:

```
Training Set:    26,898,226 / 30 ≈ 896,607 effective samples   ✓✓✓
Validation Set:   6,905,795 / 30 ≈ 230,193 effective samples   ✓✓✓
Test Set:         7,333,005 / 30 ≈ 244,433 effective samples   ✓✓✓
OOT Set:         13,737,152 / 30 ≈ 457,905 effective samples   ✓✓✓
```

Even after accounting for rolling window dependencies, XGBoost still has **~230K effective independent samples** in the validation set—massively more than the 7 engines available to LSTM.

## Comparative View

| Split | LSTM's Perspective | XGBoost Raw Rows | XGBoost Effective Samples* |
|-------|-------------------|------------------|---------------------------|
| **Training** | 39 sequences (sufficient) | 26.9M rows | ~897K samples ✓✓✓ |
| **Validation** | 7 sequences (insufficient) | 6.9M rows | ~230K samples ✓✓✓ |
| **Test** | 9 sequences (insufficient) | 7.3M rows | ~244K samples ✓✓✓ |
| **OOT** | 19 sequences (insufficient) | 13.7M rows | ~458K samples ✓✓✓ |

*Effective samples calculated as `rows / 30` to account for rolling window dependencies

## Key Differences Explained

### 1. Data Modeling Paradigm

**LSTM:**

```
Engine 1: [t₁, t₂, t₃, ..., t_n] → One sequence
Engine 2: [t₁, t₂, t₃, ..., t_m] → One sequence
...
Total: N engines = N sequences
```

**XGBoost (with rolling windows):**

```
Engine 1: [t₁, t₂, ..., t₃₀] → Features for row at t₃₀
          [t₂, t₃, ..., t₃₁] → Features for row at t₃₁ (shares 29 steps with t₃₀)
          [t₃, t₄, ..., t₃₂] → Features for row at t₃₂ (shares 29 steps with t₃₁)
          ...
Engine 2: [t₁, t₂, ..., t₃₀] → Features for row at t₃₀
          ...
Total: N engines = Millions of rows
       But ~millions/30 = hundreds of thousands of effective independent samples
```

### 2. Independence Assumption

**LSTM:**

- Explicitly models temporal correlation WITHIN each engine sequence
- Treats complete engine lifecycles as independent from each other
- Statistical power limited by number of independent engine sequences
- **Effective sample size = number of engines**

**XGBoost:**

- Does NOT model temporal correlation directly in the algorithm
- Instead, temporal patterns are captured via rolling window features (5, 15, 30 steps)
- Creates partial dependencies between consecutive rows (overlapping windows)
- Much weaker dependency than LSTM's sequential modeling
- **Effective sample size ≈ total rows / largest window size**

### 3. Statistical Power Comparison

**LSTM:**

```
Validation power = f(number of engines)
7 engines → Limited statistical power ✗
```

**XGBoost:**

```
Validation power = f(number of effective samples)
6.9M rows / 30 ≈ 230K effective samples → Massive statistical power ✓✓✓
```

**Key insight:** Even with the rolling window adjustment, XGBoost has **~33,000× more effective samples** than LSTM (230K vs 7)!

## Why Unit-Level Splitting Still Makes Sense

Even though XGBoost doesn't suffer from the same statistical limitation, **unit-level splitting is still the correct approach** because:

1. **Prevents data leakage**: Ensures no engine appears in multiple splits
2. **Tests generalization**: Model must work on completely unseen engines
3. **Realistic evaluation**: Simulates real-world deployment where you predict on new engines
4. **Maintains temporal integrity**: All time steps from one engine stay together
5. **Conservative evaluation**: Harder test than random row splitting

## Implications for Model Development

### For LSTM

- Small engine counts in validation/test sets are a real concern
- May need k-fold cross-validation at the engine level
- Performance metrics may have higher variance
- Consider expanding validation set if possible

### For XGBoost

- Even accounting for rolling window dependencies, sample sizes are excellent
- ~230K effective validation samples >> 30 required for statistical sufficiency
- Can confidently evaluate model performance on all splits
- Hyperparameter tuning is statistically robust
- Performance metrics will have low variance
- Rolling windows create weaker dependencies than LSTM sequences

## Conclusion

**XGBoost does NOT have the same statistical limitation as LSTM** when using the same data splitting strategy.

### The Nuanced Picture

While LSTM is limited by having only 7-19 engines in validation/test/OOT sets, XGBoost benefits from having hundreds of thousands of effective samples—even after accounting for rolling window dependencies.

**Corrected understanding:**

- **LSTM**: Sequence-based, limited by engine count
  - Validation: **7 independent sequences** ✗

- **XGBoost**: Row-based with rolling windows, limited by effective sample count
  - Validation: **~230K effective independent samples** ✓✓✓
  - This is calculated as `6.9M rows / 30` to account for overlapping 30-step windows

### Why the Difference Matters

The fundamental difference lies in how each algorithm handles temporal information:

1. **LSTM**: Models temporal dependencies directly in the algorithm architecture
   - Requires complete sequences
   - Each engine = one statistical unit

2. **XGBoost**: Captures temporal patterns via engineered features (rolling windows)
   - Each row gets features from a window of past observations
   - Creates partial dependencies, but much weaker than LSTM's sequential modeling
   - Effective sample size is reduced but still massive compared to LSTM

### Bottom Line

While we split by units to prevent data leakage (correct for both models), the statistical concern about "insufficient engines" primarily affects sequence-based models like LSTM. XGBoost, even with rolling window features creating dependencies, still has **33,000× more effective validation samples** than LSTM (230K vs 7).

## Critical Distinction: LSTM Sequence Length vs XGBoost Rolling Windows

### Both Use 30 Time Steps—But Very Differently!

**LSTM**: `SEQUENCE_LENGTH = 30` (lookback window)
**XGBoost**: Rolling windows of 5, 15, and 30 steps

Despite both using 30-step windows, they process data fundamentally differently:

### LSTM's Sequence Length

```python
SEQUENCE_LENGTH = 30  # Lookback window for LSTM
```

- LSTM **requires 30 consecutive time steps** as input to make ONE prediction
- Uses the hidden states to model temporal dependencies across the sequence
- For an engine with 1000 cycles:
  - Can make predictions starting from cycle 30 (needs 30 previous cycles)
  - Generates ~970 predictions (cycles 30-999)
- However, all predictions for one engine are **temporally connected**
  - Prediction at t=31 uses states from t=30
  - Prediction at t=32 uses states from t=31, which used states from t=30, etc.
- This creates a **dependency chain** across the entire engine lifecycle

### XGBoost's Rolling Windows

```python
windows = {'short': 5, 'medium': 15, 'long': 30}
```

- XGBoost uses rolling windows to **compute features** at each time step
- Each row gets its own set of rolling statistics (mean, std) from past observations
- For an engine with 1000 cycles:
  - Can make predictions for nearly all 1000 rows (using `min_periods=1`)
  - Each prediction uses a 30-step window of features
- Consecutive predictions share overlapping windows but **no hidden states**
  - Prediction at t=31 uses windows [t-29:t+1]
  - Prediction at t=32 uses windows [t-28:t+2]
  - They overlap but are evaluated independently by the tree model

### The Key Difference

| Aspect | LSTM (Sequence Length) | XGBoost (Rolling Windows) |
|--------|------------------------|---------------------------|
| **Purpose** | Input requirement for RNN | Feature engineering |
| **Dependencies** | Hidden states propagate through entire sequence | Only within window overlap |
| **Temporal modeling** | Explicit (in architecture) | Implicit (in features) |
| **Independence** | Entire engine is one connected sequence | Weak dependencies between rows |
| **Statistical unit** | Engine (all predictions connected) | Row (predictions quasi-independent) |

### Why This Matters for Sample Size

Even though both use 30-step windows:

**LSTM**: The entire engine lifecycle forms one continuous sequence
- Validation set: **7 complete sequences** (7 engines)
- All predictions within an engine are part of the same temporal chain

**XGBoost**: Each row is a separate prediction with overlapping features
- Validation set: **~230K quasi-independent predictions** (6.9M / 30)
- Overlapping windows create correlation but not sequential dependency

## Technical Deep Dive: Rolling Window Implementation

### How Rolling Windows Work in This Implementation

From `scripts/utils/data_processing_gold_feature_xgboost.py`:

```python
windows = {
    'short': 5,      # Last 5 cycles
    'medium': 15,    # Last 15 cycles
    'long': 30       # Last 30 cycles
}

# For each sensor, calculate rolling statistics
mean_col = grouped[sensor].transform(
    lambda x: x.rolling(window=window_size, min_periods=1).mean()
)

std_col = grouped[sensor].transform(
    lambda x: x.rolling(window=window_size, min_periods=1).std()
)
```

### Features Created (per sensor):

- **12 original sensors** (raw values at time t)
- **36 rolling mean features** (3 windows × 12 sensors)
- **36 rolling std features** (3 windows × 12 sensors)
- **12 acceleration features** (short_mean - medium_mean)
- **Total: 96 features per observation**

### Dependency Structure

For the 30-step window at time `t`:
```
Row t uses: [t-29, t-28, ..., t-1, t]
Row t+1 uses: [t-28, t-27, ..., t, t+1]
                └─────── 29 overlapping steps ───────┘
```

This 96.7% overlap (29/30) means consecutive rows are highly correlated, but still fundamentally different from LSTM's approach where the **entire sequence** is processed as a single unit.

### Why Dividing by 30 is a Conservative Estimate

Dividing total rows by 30 assumes:
- Only every 30th observation is independent
- This accounts for the longest window (30 steps)
- It's conservative because:
  - Not all features use the 30-step window (some use 5 or 15)
  - Different engines are fully independent
  - The model can still learn from the variation within overlapping windows

### Actual Statistical Power

The actual statistical power likely lies somewhere between:
- **Lower bound**: `rows / 30` ≈ 230K (very conservative)
- **Upper bound**: `rows` ≈ 6.9M (naive, assumes full independence)
- **Realistic estimate**: Probably closer to the lower bound for safety

Even using the most conservative estimate, XGBoost has **massive statistical power** compared to LSTM's 7 sequences.

## References

- Data splitting strategy: `notebooks/01-data-splitting-strategy.md`
- Original notebook: `notebooks/N CMAPSS ML Model.ipynb`
- XGBoost feature engineering: `scripts/utils/data_processing_gold_feature_xgboost.py`
- Data pipeline architecture: `notebooks/02-data-pipeline-medallion-architecture.md`
