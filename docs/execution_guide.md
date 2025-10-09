# Execution Guide: LSTM vs XGBoost Comparison

## 📋 Complete Execution Order

### Phase 1: Setup & Data Preparation (Cells 1-12)
These cells should already be complete from your previous work.

```
✓ Cell 1:  Import libraries (now includes xgboost)
✓ Cell 2:  Config variables (HEALTHY_WINDOW removed)
✓ Cell 3:  Data loading functions
✓ Cell 4:  Load N-CMAPSS datasets
✓ Cell 5:  Data integrity checks
✓ Cell 6:  EDA - distributions
✓ Cell 7:  EDA - sensor trends
✓ Cell 8:  EDA - correlations
✓ Cell 9:  EDA - lagged correlations
✓ Cell 10: Feature selection
✓ Cell 11: Train/test split
✓ Cell 12: Create Darts TimeSeries & normalize
```

**Status**: Already executed ✓

---

### Phase 2: LSTM Training (Cells 13-16)
```
✓ Cell 13: Visualize data structure
✓ Cell 14: Train LSTM model
✓ Cell 15: Save LSTM model
✓ Cell 16: Evaluate LSTM (Test & OOT)
```

**Status**: Already executed ✓

**Output Variables Created**:
- `lstm_model` - Trained LSTM model
- `rmse_test` - LSTM RMSE on test set
- `mae_test` - LSTM MAE on test set
- `rmse_oot` - LSTM RMSE on OOT set
- `mae_oot` - LSTM MAE on OOT set

---

### Phase 3: XGBoost Implementation (NEW - Cells 17-20)
These are the new cells you need to execute.

#### 🆕 Cell 17: Time-Windowed Feature Engineering
```python
print("\n=== 13. Time-Windowed XGBoost Model ===")
# Creates multi-scale features for XGBoost
```

**What it does**:
1. Defines time windows (1, 5, 15, 30 cycles)
2. Creates rolling aggregations (mean, std, max, min, range)
3. Adds acceleration features (cross-window)
4. Applies to train/val/test/OOT splits

**Expected output**:
```
Creating multi-scale rolling window features...
Windows: {'immediate': 1, 'short': 5, 'medium': 15, 'long': 30}
Features: 12 sensors
  Processing immediate window (size=1)...
  Processing short window (size=5)...
  Processing medium window (size=15)...
  Processing long window (size=30)...
  Creating cross-window acceleration features...
✓ Feature engineering complete. New shape: (N_rows, ~268 columns)

Total features for XGBoost: 264
Original features: 12
Engineered features: 252
```

**Time**: ~30-60 seconds

**Output Variables**:
- `df_train_xgb` - Training data with engineered features
- `df_val_xgb` - Validation data with engineered features
- `df_test_xgb` - Test data with engineered features
- `df_oot_xgb` - OOT data with engineered features
- `X_train`, `y_train` - Feature matrix and labels
- `xgb_feature_cols` - List of all feature names

---

#### 🆕 Cell 18: Train XGBoost Model
```python
print("\n=== 14. Train XGBoost Model ===")
# Trains gradient boosted trees with early stopping
```

**What it does**:
1. Configures XGBoost hyperparameters
2. Trains model with validation set monitoring
3. Saves trained model to disk

**Expected output**:
```
XGBoost Configuration:
  objective: reg:squarederror
  max_depth: 6
  learning_rate: 0.05
  n_estimators: 500
  tree_method: gpu_hist
  ...

Training XGBoost model...
[0]     validation_0-rmse:XX.XXXX
[50]    validation_0-rmse:XX.XXXX
[100]   validation_0-rmse:XX.XXXX
...
[423]   validation_0-rmse:XX.XXXX  <- Best iteration (example)

✓ XGBoost training complete!
  Best iteration: 423
  Best score: XX.XXXX
✓ Model saved to 'xgboost_rul_model.pkl'
```

**Time**: ~2-5 minutes (faster with GPU)

**Output Variables**:
- `xgb_model` - Trained XGBoost model

---

#### 🆕 Cell 19: Evaluate XGBoost
```python
print("\n=== 15. XGBoost Model Evaluation ===")
# Tests model on held-out data
```

**What it does**:
1. Predicts RUL on test and OOT sets
2. Calculates RMSE and MAE metrics
3. Visualizes example prediction trajectory
4. Shows feature importance rankings

**Expected output**:
```
XGBoost Evaluation on Test Set:
Root Mean Squared Error (RMSE): XX.XXXX
Mean Absolute Error (MAE): XX.XXXX

XGBoost Evaluation on OOT Set (Datasets [2, 7]):
Root Mean Squared Error (RMSE): XX.XXXX
Mean Absolute Error (MAE): XX.XXXX

[Prediction plot for example engine]

=== Top 20 Most Important Features ===
                    feature  importance
       Core_Speed_mean_long      0.XXXX
  HPC_Outlet_Pressure_std_medium 0.XXXX
         Fan_Speed_acceleration  0.XXXX
...

[Feature importance bar chart]
```

**Time**: ~30-60 seconds

**Output Variables**:
- `rmse_test_xgb` - XGBoost RMSE on test set
- `mae_test_xgb` - XGBoost MAE on test set
- `rmse_oot_xgb` - XGBoost RMSE on OOT set
- `mae_oot_xgb` - XGBoost MAE on OOT set
- `feature_importance` - DataFrame with feature rankings

---

#### 🆕 Cell 20: Model Comparison
```python
print("\n=== 16. Model Comparison: LSTM vs XGBoost ===")
# Side-by-side performance comparison
```

**What it does**:
1. Creates comparison table with metrics
2. Calculates percentage improvements
3. Generates visualization plots (2×2 grid)
4. Summarizes key insights

**Expected output**:
```
================================================================================
MODEL PERFORMANCE COMPARISON
================================================================================
              Model  Test RMSE  Test MAE  OOT RMSE  OOT MAE  ...
     LSTM (Darts)      XX.XX     XX.XX     XX.XX    XX.XX    ...
XGBoost (Time-Windowed) XX.XX   XX.XX     XX.XX    XX.XX    ...
================================================================================

[4-panel comparison plot showing Test/OOT RMSE/MAE]

================================================================================
SUMMARY
================================================================================
✓ Best model on Test Set: [Winner] (RMSE: XX.XXXX)
✓ Best model on OOT Set: [Winner] (RMSE: XX.XXXX)

✓/⚠ XGBoost improves/degrades over LSTM on Test Set by X.XX%
✓/⚠ XGBoost improves/degrades over LSTM on OOT Set by X.XX%

================================================================================
KEY INSIGHTS
================================================================================
LSTM (Darts):
  • Sequential model capturing temporal dependencies
  • Uses 30-cycle lookback window
  • 32 hidden units, 2 layers
  • Normalized features + targets

XGBoost (Time-Windowed):
  • 264 engineered features from 12 sensors
  • Multi-scale windows: immediate, short (5), medium (15), long (30)
  • Rolling aggregations: mean, std, max, min, range, acceleration
  • Tree-based ensemble learning
================================================================================
```

**Time**: ~10 seconds

---

## 🎯 Quick Execution Plan

If you want to run everything fresh:

### Option A: Run All (Full Pipeline)
```
Execute Cells 1-20 in sequence
Total time: ~15-30 minutes (depending on data size and GPU)
```

### Option B: Resume from XGBoost (Faster)
Since you already have LSTM trained:
```
1. Verify LSTM metrics exist:
   - Check that rmse_test, mae_test, rmse_oot, mae_oot are defined
   - If not, re-run Cell 16 (LSTM evaluation)

2. Execute new cells:
   - Cell 17: Feature engineering (~1 min)
   - Cell 18: Train XGBoost (~3 min)
   - Cell 19: Evaluate XGBoost (~1 min)
   - Cell 20: Comparison (~10 sec)
   
Total time: ~5 minutes
```

---

## ⚠️ Common Execution Issues

### Issue 1: "NameError: name 'rmse_test' is not defined"
**Cause**: LSTM evaluation cell (16) not run yet
**Fix**: Execute Cell 16 before Cell 20

### Issue 2: "ModuleNotFoundError: No module named 'xgboost'"
**Cause**: XGBoost not installed
**Fix**: 
```bash
conda activate llm-gpu
pip install xgboost
```
Then restart kernel and re-run Cell 1

### Issue 3: "MemoryError during feature engineering"
**Cause**: Too many features being created
**Fix**: Reduce number of windows in Cell 17:
```python
windows = {
    'short': 5,
    'long': 30  # Keep only 2 windows instead of 4
}
```

### Issue 4: XGBoost training extremely slow
**Cause**: GPU not available or not configured
**Fix**: In Cell 18, change:
```python
'tree_method': 'hist'  # Instead of 'gpu_hist'
```

### Issue 5: "ValueError: could not convert string to float"
**Cause**: Non-numeric data in features
**Fix**: Check for NaN values:
```python
print(df_train_xgb[xgb_feature_cols].isnull().sum())
df_train_xgb = df_train_xgb.fillna(0)  # Fill NaN with 0
```

---

## 📊 Expected Results Summary

After executing all cells, you should have:

### Files Created
```
✓ darts_lstm_model.pkl       - Saved LSTM model
✓ xgboost_rul_model.pkl       - Saved XGBoost model
```

### Variables in Memory
```
LSTM:
  - lstm_model, rmse_test, mae_test, rmse_oot, mae_oot

XGBoost:
  - xgb_model, rmse_test_xgb, mae_test_xgb, rmse_oot_xgb, mae_oot_xgb
  - df_train_xgb, df_val_xgb, df_test_xgb, df_oot_xgb
  - feature_importance

Comparison:
  - comparison_df (pandas DataFrame with both models' metrics)
```

### Visualizations Generated
```
1. LSTM RUL trajectory (Cell 16)
2. XGBoost RUL trajectory (Cell 19)
3. XGBoost feature importance bar chart (Cell 19)
4. 2×2 comparison plots (Cell 20)
   - Test RMSE, Test MAE
   - OOT RMSE, OOT MAE
```

---

## 🔄 Rerun Strategy

If you need to modify and rerun:

### To change XGBoost hyperparameters:
```
1. Edit Cell 18 (xgb_params dictionary)
2. Execute Cell 18 (retrain model)
3. Execute Cell 19 (re-evaluate)
4. Execute Cell 20 (update comparison)
```

### To change feature engineering:
```
1. Edit Cell 17 (windows, aggregations)
2. Execute Cell 17 (recreate features)
3. Execute Cell 18 (retrain XGBoost)
4. Execute Cell 19 (re-evaluate)
5. Execute Cell 20 (update comparison)
```

### To retrain LSTM with new config:
```
1. Edit Cell 2 (config variables)
2. Execute Cell 14 (retrain LSTM)
3. Execute Cell 16 (re-evaluate LSTM)
4. Execute Cell 20 (update comparison)
```

---

## 📈 Performance Benchmarks

Approximate execution times (on laptop with GPU):

| Cell | Task | Time | Can Skip? |
|------|------|------|-----------|
| 1-3 | Setup | < 5s | No |
| 4 | Load data | ~30s | No (unless already in memory) |
| 5-9 | EDA | ~2 min | Yes (exploratory only) |
| 10-12 | Prepare splits | ~30s | No |
| 13 | Visualize | ~20s | Yes (exploratory only) |
| 14 | Train LSTM | ~5 min | No (unless already trained) |
| 15 | Save LSTM | < 1s | No |
| 16 | Eval LSTM | ~2 min | No |
| **17** | **Engineer features** | **~1 min** | **No** |
| **18** | **Train XGBoost** | **~3 min** | **No** |
| **19** | **Eval XGBoost** | **~1 min** | **No** |
| **20** | **Comparison** | **~10s** | **No** |

**Total fresh run**: ~15-20 minutes
**Resume from LSTM**: ~5 minutes

---

## ✅ Validation Checklist

Before considering execution complete:

- [ ] No error messages in any cell
- [ ] LSTM RMSE < 30 on test set
- [ ] XGBoost RMSE < 30 on test set
- [ ] Feature importance shows logical sensors (Core Speed, HPC Pressure, etc.)
- [ ] Comparison table shows both models side-by-side
- [ ] Both models saved to disk (.pkl files exist)
- [ ] Plots render correctly

If all checked, execution is complete! ✓

---

**Next Step**: Analyze results and determine which model performs better for your use case.
