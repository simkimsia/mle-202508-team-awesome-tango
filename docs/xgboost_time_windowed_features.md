# Time-Windowed XGBoost Implementation for RUL Prediction

## Overview
This document explains the time-windowed feature engineering approach for XGBoost, comparing it with the LSTM model for turbofan engine Remaining Useful Life (RUL) prediction.

---

## Architecture Comparison

### LSTM (Darts) Approach
```
Raw Time Series → Normalize → LSTM(seq_len=30) → RUL Prediction
                                    ↓
                         Learns temporal patterns
                         automatically from sequences
```

**Key Characteristics:**
- **Sequential Learning**: LSTM processes 30-cycle sequences in order
- **Hidden State**: Maintains internal memory across time steps
- **Automatic Feature Learning**: Discovers temporal patterns without manual engineering
- **Input**: Normalized sensor readings (12 features × 30 timesteps)
- **Parameters**: ~27,000 trainable parameters (with HIDDEN_DIM=32)

---

### XGBoost (Time-Windowed) Approach
```
Raw Time Series → Multi-Scale Rolling Features → XGBoost Trees → RUL Prediction
                            ↓
                  Manual feature engineering
                  at multiple time scales
```

**Key Characteristics:**
- **Explicit Time Windows**: Creates features at 4 time scales (1, 5, 15, 30 cycles)
- **Aggregations**: Mean, std, max, min, range per window
- **Cross-Window Features**: Acceleration (change between windows)
- **Input**: ~240 engineered features from 12 sensors
- **Learning**: Tree-based ensemble (500 trees with depth=6)

---

## Feature Engineering Details

### Time Windows
| Window Name | Size | Real-World Meaning | Purpose |
|-------------|------|-------------------|---------|
| `immediate` | 1 cycle | Current snapshot | Capture instantaneous state |
| `short` | 5 cycles | Last week | Detect rapid changes |
| `medium` | 15 cycles | 2-3 weeks | Mid-term trends |
| `long` | 30 cycles | 1 month | Long-term degradation (matches LSTM) |

### Feature Types Per Window

For each sensor (12 total) × each window (4 total) = 48 base features per aggregation

1. **`sensor_mean_window`**: Average value
   - Smooth out noise
   - Capture trend level

2. **`sensor_std_window`**: Standard deviation
   - Measure variability
   - Detect instability

3. **`sensor_max_window`**: Maximum value
   - Identify peak stress
   - Detect anomalies

4. **`sensor_min_window`**: Minimum value
   - Baseline performance
   - Detect degradation floor

5. **`sensor_range_window`**: max - min
   - Operating range width
   - Variability indicator

### Cross-Window Features

**Acceleration Features**: `sensor_acceleration = mean_short - mean_medium`
- 12 sensors × 1 acceleration = 12 features
- **Purpose**: Rate of change between time scales
- **Interpretation**: Positive = getting worse faster, Negative = stabilizing

---

## Feature Count Breakdown

```
Original sensors:           12
Windows:                    4 (immediate, short, medium, long)
Aggregations per window:    5 (mean, std, max, min, range)
Cross-window:               1 (acceleration)

Base features:       12 original sensors
Window features:     12 sensors × 4 windows × 5 aggregations = 240
Acceleration:        12 sensors × 1 = 12
─────────────────────────────────────────────────
Total XGBoost features: ~264 features
```

---

## Implementation Adapted from PySpark

### Original Code (PySpark)
```python
# Window specification for Spark DataFrame
window_spec = Window.partitionBy("engine_id").orderBy("cycle").rowsBetween(-window_size, 0)

# Apply aggregations
result_df = result_df.withColumn(
    f"{sensor}_mean_{window_name}",
    avg(col(sensor)).over(window_spec)
)
```

### Adapted Code (Pandas)
```python
# Group by engine unit
grouped = result_df.groupby('unit')

# Apply rolling aggregations per group
result_df[f"{sensor}_mean_{window_name}"] = grouped[sensor].transform(
    lambda x: x.rolling(window=window_size, min_periods=1).mean()
)
```

**Key Changes:**
1. `Window.partitionBy("engine_id")` → `groupby('unit')`
2. `.over(window_spec)` → `.transform(lambda x: x.rolling(...))`
3. `min_periods=1` ensures early cycles still get features (uses available data)

---

## Why This Approach Works for RUL

### Problem Characteristics
- **Non-linear degradation**: Engines degrade slowly, then rapidly fail
- **Multi-scale patterns**: Different sensors degrade at different rates
- **Operational variations**: Flight conditions affect sensor readings

### XGBoost Advantages
1. **Explicit Multi-Scale**: Captures patterns at different time horizons
2. **Feature Interactions**: Trees learn complex sensor combinations
3. **Robustness**: Less sensitive to scaling issues
4. **Interpretability**: Feature importance shows which sensors/windows matter

### LSTM Advantages
1. **Automatic Learning**: No manual feature engineering
2. **Long-term Memory**: Can capture dependencies beyond 30 cycles
3. **Sequence Modeling**: Native understanding of temporal order
4. **Transfer Learning**: Pre-trained representations

---

## Training Configuration

### XGBoost Hyperparameters
```python
{
    'objective': 'reg:squarederror',  # Regression task
    'max_depth': 6,                   # Tree depth (prevent overfitting)
    'learning_rate': 0.05,            # Step size (conservative)
    'n_estimators': 500,              # Number of trees
    'subsample': 0.8,                 # Row sampling (regularization)
    'colsample_bytree': 0.8,          # Feature sampling per tree
    'min_child_weight': 3,            # Minimum samples per leaf
    'gamma': 0.1,                     # Pruning threshold
    'reg_alpha': 0.1,                 # L1 regularization
    'reg_lambda': 1.0,                # L2 regularization
    'early_stopping_rounds': 20,      # Stop if no improvement
    'tree_method': 'gpu_hist'         # GPU acceleration (if available)
}
```

### LSTM Configuration (for comparison)
```python
{
    'input_chunk_length': 30,         # Sequence length
    'hidden_dim': 32,                 # LSTM units
    'n_rnn_layers': 2,                # Number of layers
    'batch_size': 256,                # Training batch size
    'learning_rate': 0.001,           # Optimizer learning rate
    'epochs': 50,                     # Maximum epochs
    'early_stopping_patience': 2      # Stop after 2 epochs without improvement
}
```

---

## Expected Performance Characteristics

### XGBoost Strengths
- **Short-term accuracy**: Excellent at immediate predictions
- **Feature importance**: Clear insights into which sensors matter
- **Training speed**: Faster than LSTM (no backpropagation)
- **Interpretability**: Tree structure is explainable

### XGBoost Weaknesses
- **Fixed window**: Cannot adapt window sizes
- **Feature explosion**: 264 features (memory intensive)
- **Manual engineering**: Requires domain knowledge
- **No sequence memory**: Each prediction is independent

### LSTM Strengths
- **Long-range dependencies**: Can learn patterns across many cycles
- **Automatic features**: No manual engineering needed
- **Adaptive**: Learns optimal "windows" implicitly
- **Generalization**: Better on unseen operational conditions

### LSTM Weaknesses
- **Training time**: Slower convergence
- **Hyperparameter sensitivity**: Architecture matters
- **Black box**: Harder to interpret
- **Data hungry**: Needs more examples to learn well

---

## Evaluation Metrics

Both models compared on:

1. **Test Set**: Internal holdout units (15% of training datasets)
   - Same operational conditions as training
   - Tests interpolation capability

2. **OOT Set**: Datasets 2 & 7 (completely unseen)
   - Different operational conditions
   - Tests generalization/extrapolation

### Metrics Used
- **RMSE** (Root Mean Squared Error): Penalizes large errors heavily
- **MAE** (Mean Absolute Error): Average absolute prediction error

---

## Model Selection Guidance

### Choose XGBoost if:
- ✅ Interpretability is critical (feature importance analysis)
- ✅ Training time is limited
- ✅ You have domain knowledge for feature engineering
- ✅ Short-term predictions are more important

### Choose LSTM if:
- ✅ Long-term dependencies matter
- ✅ Generalization to new conditions is critical
- ✅ You want end-to-end learning (no manual features)
- ✅ You have sufficient training data and compute

### Ensemble Both if:
- 🌟 Maximum accuracy is needed
- 🌟 You can afford computational cost
- 🌟 Different strengths complement each other

---

## Feature Importance Analysis

After training, XGBoost provides feature importance scores showing:
- **Which sensors** are most predictive
- **Which time windows** matter most
- **Which aggregations** (mean vs std vs range) are key

Example insights:
```
Top features:
1. Core_Speed_mean_long       (30-cycle average)
2. HPC_Outlet_Pressure_std_medium  (15-cycle variability)
3. Fan_Speed_acceleration      (rate of change)
```

This tells us:
- Core speed long-term average is most important
- HPC pressure variability over medium term matters
- Fan speed acceleration (rapid changes) is a warning sign

---

## Code Structure

### Cell 13: Feature Engineering
- `create_multiscale_features()` function
- Apply to train/val/test/OOT splits
- Create feature matrix for XGBoost

### Cell 14: Model Training
- XGBoost hyperparameter configuration
- Fit with early stopping on validation set
- Save trained model

### Cell 15: Model Evaluation
- Predictions on test and OOT sets
- Calculate RMSE and MAE
- Visualize example predictions
- Plot feature importance

### Cell 16: Model Comparison
- Side-by-side comparison: LSTM vs XGBoost
- Performance metrics table
- Visualization plots
- Summary insights

---

## References

1. **Original PySpark Implementation**: Multi-scale window feature engineering
2. **N-CMAPSS Dataset**: NASA's turbofan degradation simulation
3. **Darts Library**: Time series forecasting with LSTM
4. **XGBoost**: Gradient boosting for regression

---

## Next Steps

### Potential Improvements

1. **Hyperparameter Tuning**
   - Grid search for optimal window sizes
   - Tune XGBoost depth, learning rate
   - Optimize LSTM hidden dimensions

2. **Feature Selection**
   - Remove low-importance features
   - Reduce from 264 to top 50-100
   - Speed up training

3. **Ensemble Methods**
   - Weighted average of LSTM + XGBoost
   - Stacking: Use LSTM output as XGBoost feature
   - Boosting: Iteratively improve predictions

4. **Advanced Features**
   - Sensor interaction terms (e.g., pressure × temperature)
   - Frequency domain features (FFT)
   - Physics-informed features (efficiency ratios)

---

## Summary

This implementation provides a fair comparison between:
- **LSTM**: End-to-end sequence learning
- **XGBoost**: Explicit time-windowed feature engineering

Both approaches have merit, and the best choice depends on your specific requirements for accuracy, interpretability, training time, and generalization.
