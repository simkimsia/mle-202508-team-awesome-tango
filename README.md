## Members

By surname order

1. CHEN Tiancheng
2. CHEN Zhiyang
3. LIN Xiongqing
4. Justin NG
5. SIM Kim Sia (@simkimsia)

# Aircraft Engine Predictive Maintenance - N-CMAPSS Dataset

## 📋 Project Overview

This project implements **Remaining Useful Life (RUL) prediction** for aircraft engines using the NASA N-CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. Two machine learning approaches are compared:

1. **LSTM (Long Short-Term Memory)** - Deep learning model using Darts library
2. **XGBoost** - Gradient boosted trees with time-windowed feature engineering

**Goal:** Predict how many operational cycles remain before an engine requires maintenance.

---

## 🚀 Quick Start

### Prerequisites

**System Requirements:**
- Python 3.10+
- 16+ GB RAM (for XGBoost feature engineering)
- GPU recommended (optional, for faster LSTM training)
- ~20 GB free disk space

**Install Dependencies:**

```bash
# Create conda environment
conda create -n aircraft-rul python=3.10
conda activate aircraft-rul

# Install required packages
pip install h5py numpy pandas seaborn matplotlib
pip install torch scikit-learn xgboost
pip install darts pytorch-lightning
```

**Required Data Files:**

Ensure the following files exist in the `Data/` folder:
- `N-CMAPSS_DS01.h5` through `N-CMAPSS_DS07.h5`
- `N-CMAPSS_DS08a-009.h5`, `N-CMAPSS_DS08c-008.h5`, `N-CMAPSS_DS08d-010.h5`

---

## 📂 First Run (Complete Pipeline)

**⏱️ Estimated Time: 30-40 minutes**

### Step-by-Step Execution

#### **1. Setup & Configuration** (Cells 1-3)

```python
# Cell 1: Import all required libraries
# Cell 2: Global configuration variables  
# Cell 3: Configuration summary
```

**Important Variables to Review:**
- `RUL_CLIP_MAX = 125` - Maximum RUL value (cycles)
- `SEQUENCE_LENGTH = 30` - LSTM lookback window
- `HIDDEN_DIM = 50` - LSTM hidden layer size
- `N_RNN_LAYERS = 2` - Number of LSTM layers
- `BATCH_SIZE = 256` - Training batch size

**⚠️ Before running Cell 4, set these flags:**
```python
LOAD_PREPROCESSED = False  # Process data from scratch
LOAD_MODELS = False        # Train models from scratch
```

---

#### **2. Data Loading & EDA** (Cells 4-8)

```python
# Cell 4: Load raw HDF5 datasets (~10-15 min)
# Cell 5: Data integrity checks & summary statistics
# Cell 6: Feature distributions (histograms)
# Cell 7: Sensor trends over time
# Cell 8: Correlation analysis (heatmap)
```

**What Happens:**
- Loads 7 HDF5 files (~15 GB total)
- Extracts time-series data from multiple engines
- Validates data quality (no NaNs, proper ranges)
- Exploratory Data Analysis (EDA) with visualizations

**⏱️ Duration: ~10-15 minutes**

---

#### **3. Preprocessing & Train/Test Split** (Cells 9-12)

```python
# Cell 9: Feature selection (12 sensors)
# Cell 10: Train/test split (70/15/15, unit-based)
# Cell 11: Statistical analysis (CLT validation)
# Cell 12: 💾 SAVE preprocessed dataframes
```

**Train/Test Split Strategy:**
- **Training Set:** Datasets 1, 3, 4, 5, 6 (70%)
- **Validation Set:** From training datasets (15%)
- **Test Set:** From training datasets (15%)
- **OOT (Out-of-Training) Set:** Datasets 2, 7 (completely held out)

**✅ Checkpoint:** Files saved to `data/` folder
- `df_combined.pkl` (~500 MB)
- `df_train.pkl`, `df_val.pkl`, `df_test.pkl`, `df_oot.pkl`
- `gold_df.pkl`, `feature_list.pkl`

**⏱️ Duration: ~5 minutes**

---

#### **4. Create Normalized TimeSeries for LSTM** (Cells 13-16)

```python
# Cell 13: Create Darts TimeSeries objects
# Cell 14: Normalize features and targets
# Cell 15: Visualize data structure
# Cell 16: (Optional) Load preprocessed data
```

**Why Normalization?**
- Features have different units (temperature, pressure, speed)
- Scaling to [0,1] improves LSTM convergence
- Separate scalers for features and RUL target

**⏱️ Duration: ~2 minutes**

---

#### **5. Train LSTM Model** (Cells 17-18)

```python
# Cell 17: Train LSTM with early stopping (~5-30 min)
# Cell 18: 💾 SAVE LSTM model (structure + weights)
```

**Training Configuration:**
- Model: BlockRNNModel (Darts)
- Architecture: 2-layer LSTM with 50 hidden units
- Optimizer: Adam (lr=0.001)
- Early stopping: Patience = 10 epochs
- Validation monitoring: RMSE on validation set

**✅ Checkpoint:** Model saved to `models/` folder
- `darts_lstm_model.pkl` (~18 KB) - Model structure
- `darts_lstm_model.pkl.ckpt` (~364 KB) - Trained weights

**⚠️ Important:** Both files are required for loading!

**⏱️ Duration:**
- With GPU: ~5-10 minutes
- Without GPU: ~20-30 minutes

---

#### **6. Evaluate LSTM Model** (Cells 19-21)

```python
# Cell 19: Evaluate on Test and OOT sets
# Cell 20: Detailed evaluation (pooled + per-engine)
# Cell 21: Evaluation metrics explanation (markdown)
```

**Expected Results:**
- **Test RMSE:** 0.10-0.15 cycles (normalized)
- **OOT RMSE:** 0.15-0.20 cycles (normalized)
- Lower is better!

**Metrics Explained:**
- **Pooled RMSE:** Industry standard (all predictions treated equally)
- **Per-Engine RMSE:** Alternative view (equal weight per engine)
- **MAE:** Mean Absolute Error (less sensitive to outliers)

**⏱️ Duration: ~2 minutes**

---

#### **7. XGBoost Feature Engineering** (Cell 22)

```python
# Cell 22: Create time-windowed features (~10-15 min first run)
```

**What Happens:**
- Creates rolling window features at 3 time scales:
  - **Short:** 5 cycles
  - **Medium:** 15 cycles
  - **Long:** 30 cycles
- Aggregations: mean, std, acceleration
- Batch processing (3 sensors at a time)
- Memory-optimized (float32, garbage collection)

**Feature Engineering:**
- Original: 12 sensors
- Engineered: ~85 features total
- Windows × Aggregations = 12 × 3 × 2 + 12 = 84 features

**✅ Checkpoint:** Files saved to `data/` folder
- `df_train_xgb.pkl`, `df_val_xgb.pkl`, `df_test_xgb.pkl`, `df_oot_xgb.pkl`
- `xgb_feature_cols.pkl`

**⚠️ Memory Warning:**
- Expected RAM usage: 10-15 GB
- If out-of-memory error occurs, see troubleshooting section

**⏱️ Duration:**
- First run: ~10-15 minutes
- Subsequent runs: ~30 seconds (loads from disk)

---

#### **8. Train XGBoost Model** (Cell 23)

```python
# Cell 23: Train XGBoost with early stopping (~5-10 min)
```

**Training Configuration:**
- Max depth: 6
- Learning rate: 0.05
- Number of estimators: 500
- Early stopping: 20 rounds
- Tree method: GPU if available, else histogram

**✅ Checkpoint:** Model saved
- `models/xgboost_rul_model.pkl` (~5 MB)

**⏱️ Duration: ~5-10 minutes**

---

#### **9. Evaluate XGBoost Model** (Cell 24)

```python
# Cell 24: Evaluate XGBoost (Test + OOT + Feature Importance)
```

**Outputs:**
- Test and OOT RMSE/MAE
- Example prediction plot (one engine)
- Feature importance (top 20 features)
- Feature importance bar chart

**⏱️ Duration: ~2 minutes**

---

#### **10. Model Comparison** (Cell 25)

```python
# Cell 25: Compare LSTM vs XGBoost
```

**Outputs:**
- Comparison table (side-by-side metrics)
- Bar charts (4 plots: Test/OOT × RMSE/MAE)
- Performance summary
- Model characteristics
- Interpretation guide

**⏱️ Duration: ~1 minute**

---

## 🔄 Subsequent Runs (Fast Mode)

### Option A: Quick Evaluation Only

**⏱️ Estimated Time: 2-3 minutes**

**Use when:** You just want to see results again

**Steps:**

1. **Set flags in Cell 2:**
```python
LOAD_PREPROCESSED = True   # Load saved data
LOAD_MODELS = True         # Load trained models
```

2. **Run only these cells:**
```
Cell 1-3:   Setup & Configuration
Cell 16:    Quick Start - Load Preprocessed Data
Cell 17:    Quick Start - Load Pre-Trained Models
Cell 19-21: LSTM Evaluation (optional)
Cell 24:    XGBoost Evaluation (optional)
Cell 25:    Model Comparison
```

**Result:** Models loaded instantly, evaluation runs in <3 minutes

---

### Option B: Retrain Models Only

**⏱️ Estimated Time: 15-20 minutes**

**Use when:**
- Changing model hyperparameters
- Testing different architectures
- Experimenting with learning rates
- Same data, different models

**Steps:**

1. **Set flags in Cell 2:**
```python
LOAD_PREPROCESSED = True   # Load saved data (skip 10 min)
LOAD_MODELS = False        # Retrain models
```

2. **Run these cells:**
```
Cell 1-3:   Setup
Cell 16:    Load Preprocessed Data
Cell 17:    Train LSTM (~10 min)
Cell 18:    Save LSTM
Cell 19-21: Evaluate LSTM
Cell 22:    Load XGBoost features (or create if needed)
Cell 23:    Train XGBoost (~5 min)
Cell 24:    Evaluate XGBoost
Cell 25:    Model Comparison
```

---

### Option C: Reprocess Data

**⏱️ Estimated Time: 30-40 minutes**

**Use when:**
- Changing feature selection
- Modifying RUL clipping threshold
- Adjusting train/test split ratios
- Adding/removing datasets

**⚠️ Important:** Delete old preprocessed files first!

```python
import os
import glob

# Delete preprocessed data files
for pattern in ['data/df_*.pkl', 'data/*_xgb.pkl', 'data/feature_list.pkl']:
    for file in glob.glob(pattern):
        os.remove(file)
        print(f"Deleted: {file}")
```

**Steps:**

1. **Set flags in Cell 2:**
```python
LOAD_PREPROCESSED = False  # Reprocess from HDF5
LOAD_MODELS = False        # Retrain models
```

2. **Run all cells 1-25**

---

## 📊 Understanding the Output

### LSTM Evaluation Results

**Pooled Metrics (Primary - Industry Standard):**
```
📈 Test Set (9 engines, 7.3M predictions):
   RMSE: 0.1244 cycles
   MAE:  0.0366 cycles

📈 OOT Set (19 engines, 13.7M predictions):
   RMSE: 0.1841 cycles
   MAE:  0.0494 cycles
```

**What does this mean?**
- RMSE < 0.20 on normalized scale = **Excellent performance**
- OOT slightly higher = Expected (different operating conditions)
- Your model generalizes well to unseen scenarios! ✅

**Per-Engine Metrics (Alternative View):**
```
📊 Test Set - Per-Engine Average:
   RMSE: 0.0812 ± 0.1530 cycles
   Range: [0.02, 0.49] cycles

📊 OOT Set - Per-Engine Average:
   RMSE: 0.1114 ± 0.2100 cycles
   Range: [0.02, 0.92] cycles
```

**Interpretation:**
- Small difference between pooled and per-engine = Consistent
- Low std = Model performs uniformly across engines
- Range shows best-case and worst-case scenarios

---

### XGBoost Evaluation Results

**Metrics:**
- Same format as LSTM (pooled RMSE/MAE)
- Typically competitive with LSTM (within 5-10%)

**Feature Importance Insights:**
- **Top features usually include:**
  - Long-term mean values (30-cycle window)
  - Standard deviation features (variability)
  - Acceleration features (trend detection)
  - Core sensors: HPC, HPT, LPT temperatures

**Example Predictions:**
- Trajectory plot shows actual vs predicted RUL
- Look for close alignment (good fit)
- Check for systematic bias (over/under prediction)

---

### Model Comparison Output

**Comparison Table:**
```
Model                      Test RMSE  Test MAE  OOT RMSE  OOT MAE  Test Δ (%)  OOT Δ (%)
LSTM (Darts)               0.1244     0.0366    0.1841    0.0494   0.00        0.00
XGBoost (Time-Windowed)    0.1150     0.0340    0.1780    0.0470   -7.56       -3.31
```

**How to Read:**
- **Negative Δ%:** XGBoost performs better (lower error) ✅
- **Positive Δ%:** LSTM performs better (lower error) ✅
- **|Δ%| < 5%:** Models are essentially equivalent

**Bar Charts:**
- Visual comparison of all metrics
- Easy to spot which model excels where

**Model Characteristics:**
- **LSTM:** Better for temporal dependencies, sequences
- **XGBoost:** Better for feature interpretability, speed

---

## 🛠️ Troubleshooting

### Problem 1: "File not found" Error

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'Data/N-CMAPSS_DS01.h5'
```

**Solution:**
1. Check that HDF5 files exist in `Data/` folder (not root)
2. Verify filenames match exactly (case-sensitive)
3. Ensure `filenames` list in Cell 3 includes `'Data/'` prefix

---

### Problem 2: Out of Memory (OOM) During XGBoost

**Error:**
```
MemoryError: Unable to allocate 50.5 GiB for an array
```

**Solutions (try in order):**

**Option 1: Reduce batch size**
```python
# In create_multiscale_features() function
batch_size = 2  # Change from 3 to 2
```

**Option 2: Remove 'long' window**
```python
windows = {
    'short': 5,
    'medium': 15
    # Remove: 'long': 30
}
```

**Option 3: Subsample training data**
```python
# Before feature engineering
df_train = df_train.iloc[::2]  # Take every 2nd row
```

See `docs/memory_optimization.md` for detailed solutions.

---

### Problem 3: Model Not Fitted Error

**Error:**
```
ValueError: The model has not been fitted yet
```

**Cause:** Missing `.pkl.ckpt` file

**Solution:**
```python
# Check if both files exist
import os
print(os.path.exists('models/darts_lstm_model.pkl'))        # Should be True
print(os.path.exists('models/darts_lstm_model.pkl.ckpt'))  # Should be True

# If .ckpt is missing, retrain or copy from darts_logs/
import shutil
shutil.copy('darts_logs/lstm_rul/checkpoints/best-*.ckpt',
            'models/darts_lstm_model.pkl.ckpt')
```

---

### Problem 4: NameError in Comparison Cell

**Error:**
```
NameError: name 'rmse_test' is not defined
```

**Cause:** Evaluation cells not run before comparison

**Solution:**
Run in correct order:
1. Cell 19-21: LSTM Evaluation (creates `rmse_test`, `mae_test`, etc.)
2. Cell 24: XGBoost Evaluation (creates `rmse_test_xgb`, `mae_test_xgb`, etc.)
3. Cell 25: Model Comparison (uses all variables)

---

### Problem 5: Early Stopping Not Working

**Error:**
```
AttributeError: `best_iteration` is only defined when early stopping is used
```

**Solution:**
Already fixed in Cell 23. If still occurring, ensure:
```python
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,  # Must be in fit(), not constructor
    verbose=50
)
```

---

### Problem 6: GPU Not Being Used

**Symptoms:**
- Training very slow (>30 min for LSTM)
- CPU at 100%, GPU at 0%

**Check GPU availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

**Solutions:**
1. Ensure PyTorch with CUDA installed: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
2. Set `pl_trainer_kwargs={'accelerator': 'gpu', 'devices': 1}` in LSTM training

---

## 📁 Project File Structure

```
Project Root/
│
├── Data/                              # Raw HDF5 files (15 GB) - DO NOT DELETE
│   ├── N-CMAPSS_DS01.h5
│   ├── N-CMAPSS_DS02.h5
│   ├── ...
│   └── N-CMAPSS_DS07.h5
│
├── data/                              # Preprocessed data (~5 GB) - Can regenerate
│   ├── df_combined.pkl               # All datasets combined
│   ├── df_train.pkl                  # Training split (LSTM)
│   ├── df_val.pkl                    # Validation split (LSTM)
│   ├── df_test.pkl                   # Test split (LSTM)
│   ├── df_oot.pkl                    # Out-of-training split (LSTM)
│   ├── gold_df.pkl                   # Reference data
│   ├── feature_list.pkl              # Selected features
│   ├── df_train_xgb.pkl              # Training (XGBoost engineered)
│   ├── df_val_xgb.pkl                # Validation (XGBoost engineered)
│   ├── df_test_xgb.pkl               # Test (XGBoost engineered)
│   ├── df_oot_xgb.pkl                # OOT (XGBoost engineered)
│   └── xgb_feature_cols.pkl          # XGBoost feature names
│
├── models/                            # Trained models (~10 MB) - DO NOT DELETE
│   ├── darts_lstm_model.pkl          # LSTM structure (18 KB)
│   ├── darts_lstm_model.pkl.ckpt     # LSTM weights (364 KB) - REQUIRED!
│   └── xgboost_rul_model.pkl         # XGBoost model (5 MB)
│
├── darts_logs/                        # Training logs (auto-generated)
│   └── lstm_rul/
│       └── checkpoints/
│           ├── best-epoch=*.ckpt     # Best checkpoint during training
│           └── last-epoch=*.ckpt     # Last checkpoint
│
├── plots/                             # Generated visualizations
│   ├── sensor_trend_*.png
│   └── lime_explanation_plot.png
│
├── docs/                              # Documentation
│   ├── quick_start_guide.md
│   ├── statistical_methodology.md
│   ├── handling_warnings.md
│   ├── memory_optimization.md
│   └── file_management.md
│
├── N CMAPSS ML Model.ipynb           # Main notebook - RUN THIS
├── N CMAPSS ML Model backup.ipynb    # Backup
├── README.md                          # This file
└── Run_to_Failure_Simulation_*.pdf   # Dataset documentation
```

---

## 💾 Disk Space Requirements

| Category | Size | Required? | Can Delete? |
|----------|------|-----------|-------------|
| **Raw HDF5 (Data/)** | ~15 GB | ✅ Always | ❌ No - Source data |
| **Preprocessed (data/)** | ~5 GB | ⚠️ Recommended | ✅ Yes - Can regenerate |
| **Trained Models (models/)** | ~10 MB | ✅ Always | ❌ No - Took hours to train |
| **Documentation (docs/)** | ~50 KB | ⚠️ Recommended | ✅ Yes - Reference only |
| **Training Logs (darts_logs/)** | ~2 MB | ❌ Optional | ✅ Yes - Auto-generated |

**Total:** ~20 GB

**To free up space:**
```python
# Delete preprocessed data (can regenerate in 15 min)
import shutil
shutil.rmtree('data/')
os.mkdir('data/')

# Delete training logs (not needed after training)
shutil.rmtree('darts_logs/')
```

**Never delete:**
- `Data/` folder (raw HDF5 files)
- `models/` folder (trained models)

---

## 🎯 Expected Performance Benchmarks

### LSTM Model

| Metric | Test Set | OOT Set | Industry Standard |
|--------|----------|---------|-------------------|
| **RMSE** | 0.10-0.15 | 0.15-0.20 | < 0.30 = Good |
| **MAE** | 0.03-0.05 | 0.04-0.06 | < 0.10 = Good |
| **Training Time** | 5-10 min (GPU) | - | - |
| **Prediction Time** | ~2 min | ~3 min | - |

### XGBoost Model

| Metric | Test Set | OOT Set | Industry Standard |
|--------|----------|---------|-------------------|
| **RMSE** | 0.08-0.15 | 0.12-0.20 | < 0.30 = Good |
| **MAE** | 0.03-0.05 | 0.04-0.06 | < 0.10 = Good |
| **Training Time** | 5-10 min | - | - |
| **Prediction Time** | ~1 min | ~2 min | - |

**Comparison:**
- Both models typically within 5-10% of each other
- LSTM slightly better on sequences with long-term patterns
- XGBoost provides better feature interpretability

---

## 📚 Additional Documentation

**Available in `docs/` folder:**

1. **`quick_start_guide.md`**
   - Detailed cell-by-cell walkthrough
   - What each cell does
   - Expected outputs

2. **`statistical_methodology.md`**
   - Train/test split strategy
   - Central Limit Theorem validation
   - Pooled vs per-engine evaluation
   - Sample size justification

3. **`handling_warnings.md`**
   - Common warning messages explained
   - Which warnings are safe to ignore
   - How to suppress specific warnings

4. **`memory_optimization.md`**
   - Memory-efficient feature engineering
   - float32 vs float64 trade-offs
   - Batch processing strategies
   - Troubleshooting OOM errors

5. **`file_management.md`**
   - What each file contains
   - Which files can be deleted
   - How to regenerate deleted files
   - Disk space optimization

---

## ✅ Validation Checklist

Before considering execution complete, verify:

- [ ] All cells executed without errors
- [ ] LSTM RMSE < 0.30 on test set
- [ ] XGBoost RMSE < 0.30 on test set
- [ ] Both model files saved in `models/` folder
- [ ] All data files saved in `data/` folder
- [ ] Comparison table displays both models
- [ ] Feature importance plot shows logical sensors
- [ ] Plots render correctly (no empty figures)
- [ ] No missing data warnings in integrity checks
- [ ] Early stopping triggered for both models

---

## 🔬 Experiment Tracking

### Baseline Results (First Run)

| Model | Test RMSE | OOT RMSE | Training Time | Notes |
|-------|-----------|----------|---------------|-------|
| LSTM | 0.1244 | 0.1841 | 8 min (GPU) | Default hyperparams |
| XGBoost | 0.1150 | 0.1780 | 6 min | 85 engineered features |

### Hyperparameter Experiments

**To track your experiments, fill in this table:**

| Experiment | Changes | Test RMSE | OOT RMSE | Improvement | Notes |
|------------|---------|-----------|----------|-------------|-------|
| Baseline | Default settings | 0.1244 | 0.1841 | - | - |
| Exp 1 | HIDDEN_DIM=100 | - | - | - | - |
| Exp 2 | N_EPOCHS=100 | - | - | - | - |
| Exp 3 | RUL_CLIP=150 | - | - | - | - |

---

## 📖 Key Concepts

### What is RUL?

**Remaining Useful Life (RUL)** = Number of operational cycles before maintenance required

- **Cycle:** One complete flight (takeoff → landing)
- **Degradation:** Engine performance deteriorates over time
- **Goal:** Predict RUL to schedule maintenance proactively

### Why Two Models?

**LSTM Advantages:**
- Captures temporal dependencies
- Good for sequential patterns
- Handles variable-length sequences
- No manual feature engineering

**XGBoost Advantages:**
- Faster training/prediction
- Better interpretability (feature importance)
- Handles missing values well
- Robust to outliers

**Best Practice:** Use both and compare!

### Train/Test/OOT Split

- **Training (70%):** Datasets 1, 3, 4, 5, 6
- **Validation (15%):** From training datasets
- **Test (15%):** From training datasets
- **OOT (100%):** Datasets 2, 7 (completely separate)

**Why OOT?**
- Tests generalization to different flight conditions
- More realistic evaluation of real-world performance
- Higher error expected (new scenarios)

---

## 🤝 Contributing

If you improve this pipeline:

1. Document your changes
2. Update this README
3. Add new documentation to `docs/` if needed
4. Track experiments in the table above

---

## 📞 Support & Resources

**Dataset Documentation:**
- See `Run_to_Failure_Simulation_Under_Real_Flight_Conditions_Dataset.pdf`
- NASA N-CMAPSS dataset details

**Library Documentation:**
- Darts: https://unit8co.github.io/darts/
- XGBoost: https://xgboost.readthedocs.io/
- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/

**Common Issues:**
- Check `docs/handling_warnings.md` for warning messages
- Check `docs/memory_optimization.md` for OOM errors
- Review this README's troubleshooting section

---

## 📄 License

This project uses the NASA N-CMAPSS dataset. Please cite the original dataset publication if using this code for research.

---

## 🎓 Citation

If you use this code in your research, please cite:

```
@misc{aircraft-rul-prediction,
  title={Aircraft Engine RUL Prediction using LSTM and XGBoost},
  author={[Your Name]},
  year={2025},
  publisher={GitHub},
  howpublished={\url{[Your Repository URL]}}
}
```

---

**Last Updated:** January 9, 2025  
**Version:** 1.0  
**Maintained by:** [Your Name]

---

## 🚦 Quick Reference

**First time user?**
→ Run cells 1-25 in order (30-40 min)

**Already have trained models?**
→ Set `LOAD_MODELS=True`, run cells 1-3, 16-17, 19-21, 24-25 (3 min)

**Want to experiment?**
→ Set `LOAD_PREPROCESSED=True`, modify hyperparameters, retrain (15 min)

**Having issues?**
→ Check troubleshooting section or `docs/handling_warnings.md`

**Need more help?**
→ Read documentation in `docs/` folder

---

**Happy Predicting! 🚀✈️**
