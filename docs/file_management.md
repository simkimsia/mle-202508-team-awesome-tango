# File Management Guide: What to Keep and What to Delete

## 📁 Complete File Inventory

### **models/ Folder (Keep All)**
```
models/
├── darts_lstm_model.pkl           (~18 KB)   - LSTM model structure
├── darts_lstm_model.pkl.ckpt      (~364 KB)  - LSTM trained weights (REQUIRED!)
└── xgboost_rul_model.pkl          (~5 MB)    - XGBoost trained model
```

**Status:** ✅ **Keep all files**
- Both LSTM files required together (structure + weights)
- XGBoost file contains complete trained model

---

### **data/ Folder (Organized by Type)**

#### **1. Preprocessed Data (LSTM Input)**
```
data/
├── df_combined.pkl                (~500 MB)  - Combined all datasets
├── gold_df.pkl                    (~50 MB)   - Gold standard reference
├── df_train.pkl                   (~350 MB)  - Training split
├── df_val.pkl                     (~75 MB)   - Validation split
├── df_test.pkl                    (~75 MB)   - Test split
├── df_oot.pkl                     (~150 MB)  - Out-of-training split
└── feature_list.pkl               (~1 KB)    - Selected sensor names
```

**Status:** ✅ **Keep all - used for LSTM**
- Required if you want to skip data loading (saves 10-15 min)
- Can be deleted to force reprocessing from raw HDF5 files

---

#### **2. XGBoost Engineered Data**
```
data/
├── df_train_xgb.pkl               (~2 GB)    - Training + engineered features
├── df_val_xgb.pkl                 (~500 MB)  - Validation + engineered features
├── df_test_xgb.pkl                (~500 MB)  - Test + engineered features
├── df_oot_xgb.pkl                 (~1 GB)    - OOT + engineered features
└── xgb_feature_cols.pkl           (~2 KB)    - XGBoost feature names
```

**Status:** ✅ **Keep all - used for XGBoost**
- Saves 10-15 minutes of feature engineering
- Already includes rolling windows, aggregations, acceleration features
- Can be deleted to force recomputation

---

### **Data/ Folder (Raw HDF5 Files - Root Directory)**
```
Data/
├── N-CMAPSS_DS01.h5               (~1.5 GB)  - Raw dataset 1
├── N-CMAPSS_DS02.h5               (~1.5 GB)  - Raw dataset 2 (OOT)
├── N-CMAPSS_DS03.h5               (~1.5 GB)  - Raw dataset 3
├── N-CMAPSS_DS04.h5               (~1.5 GB)  - Raw dataset 4
├── N-CMAPSS_DS05.h5               (~1.5 GB)  - Raw dataset 5
├── N-CMAPSS_DS06.h5               (~1.5 GB)  - Raw dataset 6
├── N-CMAPSS_DS07.h5               (~1.5 GB)  - Raw dataset 7 (OOT)
├── N-CMAPSS_DS08a-009.h5          (~1.5 GB)  - Raw dataset 8a
├── N-CMAPSS_DS08c-008.h5          (~1.5 GB)  - Raw dataset 8c
└── N-CMAPSS_DS08d-010.h5          (~1.5 GB)  - Raw dataset 8d
```

**Status:** ✅ **Keep all - source data**
- Original unprocessed data
- Required if you delete preprocessed files
- Cannot be regenerated (unless you re-download)

---

## 🎯 **To Answer Your Question:**

### **After running "Save XGBoost Engineered Dataframes"**

**BEFORE (Old Code - Had Duplicates):**
- Cell 38 saves: `df_train_xgb.pkl`, `df_val_xgb.pkl`, etc.
- Cell 40 saves: Same files again (DUPLICATE!)

**AFTER (Fixed Code - No Duplicates):**
- Cell 38 saves: `df_train_xgb.pkl`, `df_val_xgb.pkl`, etc. (ONLY SAVE)
- Cell 40: Now just a markdown explaining files (NO DUPLICATE SAVE)

### **Files You Need in data/ folder:**

✅ **LSTM Data (for LSTM model):**
```
df_combined.pkl
df_train.pkl
df_val.pkl
df_test.pkl
df_oot.pkl
gold_df.pkl
feature_list.pkl
```

✅ **XGBoost Data (for XGBoost model):**
```
df_train_xgb.pkl
df_val_xgb.pkl
df_test_xgb.pkl
df_oot_xgb.pkl
xgb_feature_cols.pkl
```

---

## 🗑️ **Can You Delete Anything?**

### **Safe to Delete (Can Regenerate):**

❌ **Delete `data/df_*_xgb.pkl` files IF:**
- Want to save disk space (~4 GB)
- Willing to wait 10-15 min to regenerate
- Changing feature engineering logic

❌ **Delete `data/df_*.pkl` (non-XGB) files IF:**
- Want to save disk space (~1 GB)
- Willing to wait 10-15 min to reload from HDF5
- Changing preprocessing logic

**How to regenerate:**
- Just run the notebook cells again
- Code automatically detects missing files
- Reprocesses from raw HDF5 data

---

### **NEVER Delete (Cannot Regenerate):**

✅ **Keep `Data/N-CMAPSS_DS*.h5` files:**
- Source data from NASA
- Cannot be regenerated
- Required for all processing

✅ **Keep `models/` files:**
- Trained models (took hours to train)
- Cannot be regenerated without retraining
- Critical for making predictions

---

## 📊 **Disk Space Summary**

| Category | Size | Keep? | Why |
|----------|------|-------|-----|
| **Raw HDF5** | ~15 GB | ✅ Always | Source data (irreplaceable) |
| **Preprocessed (LSTM)** | ~1 GB | ✅ Recommended | Saves 10-15 min |
| **Engineered (XGBoost)** | ~4 GB | ✅ Recommended | Saves 10-15 min |
| **Trained Models** | ~6 MB | ✅ Always | Took hours to train |
| **Documentation** | ~50 KB | ✅ Always | Reference materials |

**Total Disk Usage:** ~20 GB

---

## 🔄 **Typical Workflow**

### **First Run (No Saved Files):**
```
1. Load raw HDF5 files (10-15 min)
   ↓
2. Preprocess & save to data/*.pkl (5 min)
   ↓
3. Train LSTM & save to models/ (30 min - 2 hours)
   ↓
4. Engineer XGBoost features & save to data/*_xgb.pkl (10-15 min)
   ↓
5. Train XGBoost & save to models/ (5-10 min)
```

**Total:** ~60-150 minutes

---

### **Subsequent Runs (With Saved Files):**
```
1. Load data/*.pkl files (30 sec)
   ↓
2. Load models/darts_lstm_model.pkl + .ckpt (5 sec)
   ↓
3. Load data/*_xgb.pkl files (30 sec)
   ↓
4. Load models/xgboost_rul_model.pkl (1 sec)
   ↓
5. Run evaluation only (2-5 min)
```

**Total:** ~3-6 minutes 🚀

---

## 💡 **Best Practices**

### **1. Keep Everything During Development**
- Disk space is cheap
- Time is valuable
- Fast iterations = better experimentation

### **2. Clean Up When Changing Logic**
Delete preprocessed/engineered files if you:
- Change feature selection
- Modify RUL clipping
- Adjust train/test split
- Change rolling window sizes

### **3. Backup Trained Models**
```bash
# Backup models before retraining
Copy-Item -Path "models\" -Destination "models_backup\" -Recurse
```

### **4. Version Control Key Files**
Git track:
- ✅ Notebook (`.ipynb`)
- ✅ Documentation (`.md`)
- ❌ Data files (too large, use `.gitignore`)
- ❌ Model files (too large, track separately)

---

## 🎓 **Summary**

**Your question:** *"Do we still need some of the XGB pickle files within the Data folder?"*

**Answer:** 

✅ **YES, keep them!** They serve different purposes:

1. **`data/df_train.pkl`** (etc.) → Input for LSTM model
   - Preprocessed, no engineered features
   - Used by Darts LSTM

2. **`data/df_train_xgb.pkl`** (etc.) → Input for XGBoost model
   - Preprocessed + engineered features (rolling windows, etc.)
   - Used by XGBoost

**They are NOT duplicates!**
- LSTM files: 12 original sensors only
- XGBoost files: 12 original + 85 engineered features

**Both are needed** if you want to run both models without re-engineering features each time.

---

## 🗂️ **Recommended File Organization**

```
Project Root/
│
├── Data/                          # Raw HDF5 files (15 GB)
│   └── N-CMAPSS_DS*.h5           # NEVER DELETE
│
├── data/                          # Processed pickle files (5 GB)
│   ├── df_*.pkl                  # LSTM preprocessed (1 GB)
│   └── df_*_xgb.pkl              # XGBoost engineered (4 GB)
│
├── models/                        # Trained models (6 MB)
│   ├── darts_lstm_model.pkl      # NEVER DELETE (structure)
│   ├── darts_lstm_model.pkl.ckpt # NEVER DELETE (weights)
│   └── xgboost_rul_model.pkl     # NEVER DELETE (trained)
│
├── docs/                          # Documentation (~50 KB)
│   ├── quick_start_guide.md
│   ├── statistical_methodology.md
│   ├── handling_warnings.md
│   └── memory_optimization.md
│
└── N CMAPSS ML Model.ipynb        # Main notebook
```

**Clean, organized, and efficient!** 🎉

