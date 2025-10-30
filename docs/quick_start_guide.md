# Quick Start Guide

This guide explains how to use the saved dataframes and models to skip preprocessing and training.

## 📁 Folder Structure

```
.
├── data/                           # Saved preprocessed dataframes
│   ├── df_combined.pkl            # Main processed dataframe
│   ├── gold_df.pkl                # Feature-selected dataframe
│   ├── df_train.pkl               # Training split
│   ├── df_val.pkl                 # Validation split
│   ├── df_test.pkl                # Test split
│   ├── df_oot.pkl                 # Out-of-time test split
│   ├── df_train_xgb.pkl          # XGBoost engineered training data
│   ├── df_val_xgb.pkl            # XGBoost engineered validation data
│   ├── df_test_xgb.pkl           # XGBoost engineered test data
│   ├── df_oot_xgb.pkl            # XGBoost engineered OOT data
│   ├── feature_list.pkl          # List of selected features
│   └── xgb_feature_cols.pkl      # List of XGBoost engineered features
│
├── models/                         # Saved trained models
│   ├── darts_lstm_model.pkl      # Trained LSTM model
│   └── xgboost_rul_model.pkl     # Trained XGBoost model
│
└── N CMAPSS ML Model.ipynb        # Main notebook
```

## 🚀 Three Ways to Run the Notebook

### Option 1: From Scratch (First Time)
**When:** Running for the first time or want to reprocess everything

**Steps:**
1. Set `LOAD_PREPROCESSED = False` in the "Quick Start: Load Preprocessed Data" cell
2. Set `LOAD_MODELS = False` in the "Quick Start: Load Pre-Trained Models" cell
3. Run all cells sequentially

**Time:** ~10-15 minutes for data loading + ~5-10 minutes for training

---

### Option 2: Skip Data Loading (Already Preprocessed)
**When:** You've already preprocessed the data once

**Steps:**
1. Set `LOAD_PREPROCESSED = True` in the "Quick Start: Load Preprocessed Data" cell
2. Run the quick start cell to load saved dataframes
3. Skip to Cell 9 (Create Darts TimeSeries) to continue

**Time:** ~5-10 seconds to load data + ~5-10 minutes for training

---

### Option 3: Skip Everything (Already Trained)
**When:** You've already trained models and just want to evaluate/compare

**Steps:**
1. Set `LOAD_PREPROCESSED = True` 
2. Set `LOAD_MODELS = True`
3. Run both quick start cells
4. Skip directly to Cell 11 (Model Evaluation) or Cell 15 (XGBoost Evaluation)

**Time:** ~10 seconds to load everything

---

## 📝 Step-by-Step Workflow

### First Run (Complete Pipeline)
```
Cell 1-2:  Import packages & Config
Cell 3-6:  Data loading from HDF5 files
Cell 7:    Feature selection
Cell 8:    Train/test split
Cell 8.1:  💾 SAVE preprocessed dataframes  ← Auto-saves here
Cell 9:    Create Darts TimeSeries & normalize
Cell 10:   Train LSTM model
           💾 SAVE LSTM model              ← Auto-saves here
Cell 11:   Evaluate LSTM model
Cell 13:   XGBoost feature engineering
Cell 13.1: 💾 SAVE XGBoost dataframes       ← Auto-saves here
Cell 14:   Train XGBoost model
           💾 SAVE XGBoost model           ← Auto-saves here
Cell 15:   Evaluate XGBoost model
Cell 16:   Compare models
```

### Quick Restart (Skip Data Loading)
```
Cell 1-2:  Import packages & Config
Cell ?:    Quick Start: Load Preprocessed Data  ← Load saved dataframes
Cell 9:    Create Darts TimeSeries & normalize
Cell 10:   Train LSTM (or load model)
Cell 11:   Evaluate LSTM
Cell 14:   Train XGBoost (or load model)
Cell 15:   Evaluate XGBoost
Cell 16:   Compare models
```

### Fastest (Evaluation Only)
```
Cell 1-2:  Import packages & Config
Cell ?:    Quick Start: Load Preprocessed Data  ← Load saved dataframes
Cell ?:    Quick Start: Load Pre-Trained Models ← Load saved models
Cell 11:   Evaluate LSTM
Cell 15:   Evaluate XGBoost
Cell 16:   Compare models
```

---

## 💡 Tips

### When to Reprocess Data
- Changed feature selection (`final_features`)
- Changed RUL clipping (`RUL_CLIP_MAX`)
- Changed train/test split ratios
- Added new datasets

### When to Retrain Models
- Changed model hyperparameters (`SEQUENCE_LENGTH`, `HIDDEN_DIM`, etc.)
- Changed feature engineering for XGBoost
- Want to experiment with different architectures

### Storage Space
- **Data folder:** ~50-100 MB (depends on dataset size)
- **Models folder:** ~10-20 MB
- Total: ~100 MB for complete saved state

---

## 🔧 Troubleshooting

### "File not found" errors
**Problem:** Saved files don't exist yet

**Solution:** Run the complete pipeline once with `LOAD_PREPROCESSED = False`

### Out of memory errors
**Problem:** Loading large dataframes

**Solution:** 
- Close other programs
- Restart kernel before loading
- Process data in smaller chunks

### Model compatibility errors
**Problem:** Model was saved with different Darts version

**Solution:** Retrain models with current environment

---

## 📊 What Gets Saved

### Basic Dataframes (Cell 8.1)
- `df_combined`: Raw combined data from all HDF5 files
- `gold_df`: Feature-selected data with RUL clipping
- `df_train/val/test/oot`: Train/validation/test/OOT splits

### XGBoost Engineered (Cell 13.1)
- `df_train/val/test/oot_xgb`: Multi-scale rolling window features
  - Immediate window (1 cycle)
  - Short window (5 cycles)
  - Medium window (15 cycles)
  - Long window (30 cycles)
  - Statistical aggregations: mean, std, max, min, range
  - Cross-window features: acceleration

### Models
- `darts_lstm_model.pkl`: LSTM with early stopping checkpoint
- `xgboost_rul_model.pkl`: XGBoost with best iteration

---

## 🎯 Recommended Workflow

1. **First time:** Run everything once to generate saved files
2. **Experiments:** Load preprocessed data, modify models, retrain
3. **Analysis:** Load everything, skip to evaluation/comparison
4. **Production:** Load models only for inference

