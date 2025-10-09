# XGBoost Time-Windowed Features - Quick Reference

## 🎯 Quick Comparison

| Aspect | LSTM (Darts) | XGBoost (Time-Windowed) |
|--------|--------------|-------------------------|
| **Input** | 12 features × 30 timesteps | ~264 engineered features |
| **Learning** | Automatic from sequences | Manual feature engineering |
| **Training Time** | Slower (backprop) | Faster (tree-based) |
| **Interpretability** | Black box | Feature importance |
| **Memory** | Recurrent states | Independent predictions |
| **Best For** | Long-term patterns | Short-term patterns |

---

## 🔧 Feature Engineering Formula

```
For each sensor (12 total):
  For each window (immediate=1, short=5, medium=15, long=30):
    ✓ mean_window   = rolling average
    ✓ std_window    = rolling standard deviation  
    ✓ max_window    = rolling maximum
    ✓ min_window    = rolling minimum
    ✓ range_window  = max_window - min_window
  
  ✓ acceleration = mean_short - mean_medium

Total: 12 sensors × (4 windows × 5 aggs + 1 accel) = 252 features
       + 12 original sensors = 264 features
```

---

## 📊 Running the Comparison

### Step 1: Install XGBoost
```python
# In your conda environment
conda activate llm-gpu
pip install xgboost
```

### Step 2: Run Cells in Order
1. **Cell 1-12**: Data loading, EDA, LSTM training (already complete)
2. **Cell 13**: 🆕 Create time-windowed features
3. **Cell 14**: 🆕 Train XGBoost model
4. **Cell 15**: 🆕 Evaluate XGBoost
5. **Cell 16**: 🆕 Compare LSTM vs XGBoost

### Step 3: Analyze Results
- Check RMSE/MAE on Test and OOT sets
- Review feature importance plot
- Compare prediction trajectories

---

## 💡 Key Insights to Look For

### From Feature Importance Plot
```
High Importance = Critical for RUL prediction
- Which sensors? (e.g., Core Speed, HPC Pressure)
- Which windows? (immediate vs long-term)
- Which aggregations? (mean vs std vs acceleration)
```

### From Performance Metrics
```
Lower RMSE on Test = Better interpolation
Lower RMSE on OOT = Better generalization

LSTM typically better at: Long-term generalization
XGBoost typically better at: Short-term accuracy, interpretability
```

---

## 🎛️ Hyperparameter Tuning Quick Guide

### XGBoost: If model is...
| Symptom | Adjust | Direction |
|---------|--------|-----------|
| Overfitting | `max_depth` | Decrease (6→4) |
| Overfitting | `learning_rate` | Decrease (0.05→0.01) |
| Underfitting | `n_estimators` | Increase (500→1000) |
| Slow training | `subsample` | Decrease (0.8→0.5) |
| Too many features | Remove low importance | Keep top 100 |

### LSTM: If model is...
| Symptom | Adjust | Direction |
|---------|--------|-----------|
| Overfitting | `HIDDEN_DIM` | Decrease (32→16) |
| Underfitting | `HIDDEN_DIM` | Increase (32→64) |
| Not learning | `LEARNING_RATE` | Increase (0.001→0.01) |
| Unstable | `BATCH_SIZE` | Increase (256→512) |

---

## 🔍 Troubleshooting

### Issue: "xgboost module not found"
```bash
conda activate llm-gpu
pip install xgboost
```

### Issue: XGBoost training very slow
```python
# Change in Cell 14:
'tree_method': 'hist'  # Instead of 'gpu_hist'
'n_jobs': 4            # Limit CPU cores
```

### Issue: Out of memory during feature engineering
```python
# Reduce window sizes in Cell 13:
windows = {
    'short': 5,
    'long': 30  # Only keep 2 windows
}
```

### Issue: LSTM metrics not found
```python
# Make sure you ran Cell 11 (LSTM evaluation) first
# Variables needed: rmse_test, mae_test, rmse_oot, mae_oot
```

---

## 📈 Expected Performance Ranges

Based on N-CMAPSS dataset characteristics:

```
LSTM (typical):
  Test RMSE:  15-25 cycles
  OOT RMSE:   20-35 cycles

XGBoost (typical):
  Test RMSE:  12-22 cycles  (often better)
  OOT RMSE:   18-40 cycles  (varies)

Note: Your actual results depend on:
- Model configuration (HIDDEN_DIM, max_depth, etc.)
- Training/validation split randomness
- Early stopping timing
```

---

## 🚀 Advanced Extensions

### 1. Ensemble Both Models
```python
# Weighted average
ensemble_pred = 0.6 * lstm_pred + 0.4 * xgb_pred
```

### 2. Use LSTM Features in XGBoost
```python
# Add LSTM hidden states as features
lstm_features = extract_hidden_states(lstm_model)
X_train_enhanced = np.hstack([X_train, lstm_features])
```

### 3. Adaptive Windows
```python
# Different windows per sensor based on importance
important_sensors = ['Core Speed', 'HPC Outlet Pressure']
windows_important = [1, 5, 15, 30, 50]  # More windows
windows_others = [5, 30]  # Fewer windows
```

---

## 📋 Checklist Before Submitting Results

- [ ] Both models trained successfully
- [ ] Test RMSE < 30 cycles for at least one model
- [ ] OOT RMSE < 40 cycles for at least one model
- [ ] Feature importance plot shows logical sensors (Core Speed, HPC Pressure, etc.)
- [ ] Prediction trajectory plot shows reasonable RUL decay
- [ ] Comparison table shows clear winner or close tie
- [ ] Documentation explains why one model performs better

---

## 🎓 Interpretation Guide

### Feature Importance Example
```
Top 5 Features:
1. Core_Speed_mean_long (30-cycle avg)
   → Long-term engine wear indicator
   
2. HPC_Outlet_Pressure_std_medium (15-cycle std)
   → Mid-term operational instability
   
3. Fan_Speed_acceleration
   → Rapid degradation signal
   
4. LPT_Coolant_Bleed_range_short (5-cycle range)
   → Short-term cooling system variability
   
5. Fuel_Flow_Ratio_mean_long (30-cycle avg)
   → Long-term efficiency decline
```

### What This Tells You
- **Long-term averages** (mean_long) = Overall degradation trend
- **Medium-term variability** (std_medium) = Instability warnings
- **Acceleration** = Rate of degradation change
- **Short-term range** = Operational volatility

---

## 📞 Quick Help

**Problem**: Models not comparable (very different RMSE)
**Solution**: Check if both use same RUL_CLIP_MAX (should be 90)

**Problem**: XGBoost much worse than LSTM
**Solution**: Increase n_estimators to 1000, tune learning_rate

**Problem**: Both models poor (RMSE > 40)
**Solution**: Check data splits, verify sensors selected correctly

**Problem**: Feature importance unclear
**Solution**: Filter to top 20 only, group by window type

---

**Last Updated**: Cell 13-16 implementation
**Documentation**: See `xgboost_time_windowed_features.md` for full details
