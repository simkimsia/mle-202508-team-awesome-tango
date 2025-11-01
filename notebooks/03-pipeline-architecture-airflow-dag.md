# Pipeline Architecture: Airflow DAG Design

This document describes the complete data pipeline architecture implemented in Airflow, following the medallion architecture (Bronze → Silver → Gold) with separate label and feature stores for LSTM and XGBoost models.

## Table of Contents

- [Overview](#overview)
- [Pipeline Flow](#pipeline-flow)
- [Task Definitions](#task-definitions)
- [Why Separate Label Stores?](#why-separate-label-stores)
- [Directory Structure](#directory-structure)
- [Data Flow Details](#data-flow-details)
- [Implementation Files](#implementation-files)

## Overview

The pipeline implements a **7-stage data processing workflow** that transforms raw N-CMAPSS HDF5 files into model-ready features for both LSTM and XGBoost models.

**Key Design Principles:**
1. **Medallion Architecture**: Bronze (raw) → Silver (cleaned) → Gold (model-specific)
2. **Separation of Concerns**: Labels and features are processed independently
3. **Model-Specific Processing**: Each model gets data in its optimal format
4. **Parallel Execution**: LSTM and XGBoost branches run in parallel after silver layer
5. **Reproducibility**: Snapshot dates ensure consistent data versioning

## Pipeline Flow

```
start_pipeline
    ↓
┌─────────────────────┐
│  bronze_ncmapss     │  Load raw HDF5 files
└─────────────────────┘
    ↓
┌─────────────────────┐
│  silver_ncmapss     │  Clean & standardize
└─────────────────────┘
    ↓
┌─────────────────────┐
│ gold_label_base     │  Create RUL_Clipped + splits
│     _ncmapss        │  (as DataFrames)
└─────────────────────┘
    ↓
    ├───────────────────────────┬───────────────────────────┐
    ↓                           ↓                           ↓
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│ gold_label   │          │ gold_label   │          │              │
│ _lstm        │          │ _xgboost     │          │              │
│ _ncmapss     │          │ _ncmapss     │          │              │
│              │          │              │          │              │
│ (TimeSeries) │          │ (DataFrame)  │          │              │
└──────────────┘          └──────────────┘          │              │
    ↓                           ↓                    │              │
┌──────────────┐          ┌──────────────┐          │              │
│ gold_feature │          │ gold_feature │          │              │
│ _lstm        │          │ _xgboost     │          │              │
│ _ncmapss     │          │ _ncmapss     │          │              │
│              │          │              │          │              │
│ (12 features)│          │ (96 features)│          │              │
└──────────────┘          └──────────────┘          │              │
    ↓                           ↓                    ↓              ↓
    └───────────────────────────┴────────────────────┴──────────────┘
                                ↓
                          end_pipeline
```

## Task Definitions

### 1. bronze_ncmapss

**Purpose:** Load raw N-CMAPSS HDF5 files into the data lake

**Script:** `scripts/bronze_ncmapss.py`

**Input:**
- 7 HDF5 files: `data/N-CMAPSS_DS01.h5` through `DS07.h5`

**Processing:**
1. Load each HDF5 file using `h5py`
2. Extract arrays: W, X_s, X_v, A, Y, T
3. Concatenate dev and test splits
4. Add dataset identifier
5. Save as parquet files

**Output:**
- Directory: `datamart/bronze/n_cmapss/snapshot_date={YYYY-MM-DD}/`
- Format: Parquet
- Shape: ~54.9M rows × 32 features
- Arrays: W (4), X_s (14), X_v (14)

**Task ID:** `bronze_ncmapss`

---

### 2. silver_ncmapss

**Purpose:** Clean, standardize, and validate data

**Script:** `scripts/silver_ncmapss.py`

**Input:**
- Bronze parquet files from previous step

**Processing:**
1. Load bronze parquet files
2. Rename columns using `COLUMN_RENAME_MAP`
3. Create composite unit identifier: `DS{dataset:02d}_{unit_orig:03d}`
4. Validate data integrity (NaN checks, variance analysis)
5. Save cleaned DataFrame

**Output:**
- Directory: `datamart/silver/n_cmapss/snapshot_date={YYYY-MM-DD}/`
- Format: Parquet
- Shape: ~54.9M rows × 38 columns
- Columns: metadata (4), operating conditions (4), physical sensors (14), virtual sensors (14), auxiliary (2)

**Task ID:** `silver_ncmapss`

---

### 3. gold_label_base_ncmapss

**Purpose:** Create RUL labels and perform train/val/test/oot splits

**Script:** `scripts/gold_label_base_ncmapss.py`

**Input:**
- Silver parquet from previous step

**Configuration:**
```python
RUL_CLIP_MAX = 90
TRAIN_SETS = [1, 3, 4, 5, 6]  # Internal datasets
OOT_SETS = [2, 7]              # Out-of-time datasets
```

**Processing:**
1. Load silver DataFrame
2. Create `RUL_Clipped = clip(Remaining Useful Life, upper=90)`
3. Split data by engine units:
   - Separate OOT sets (datasets 2 & 7)
   - Internal split: 85% train+val, 15% test
   - Train/val split: 85% train, 15% val
4. Save split DataFrames as parquet
5. Save split metadata (unit lists)

**Output:**
- Directory: `datamart/gold/label_base/n_cmapss/snapshot_date={YYYY-MM-DD}/`
- Format: Parquet (DataFrame)
- Files:
  - `df_train.parquet` (~39 engines, 26.9M rows)
  - `df_val.parquet` (~7 engines, 6.9M rows)
  - `df_test.parquet` (~9 engines, 7.3M rows)
  - `df_oot.parquet` (~19 engines, 13.7M rows)
  - `split_metadata.json` (unit lists for each split)

**Task ID:** `gold_label_base_ncmapss`

---

### 4. gold_label_lstm_ncmapss

**Purpose:** Convert labels to LSTM-compatible TimeSeries format

**Script:** `scripts/gold_label_lstm_ncmapss.py`

**Input:**
- Label base DataFrames from previous step

**Processing:**
1. Load train/val/test/oot parquet files
2. For each split:
   - Group by engine unit
   - Create Darts TimeSeries objects (one per engine)
   - Each TimeSeries contains RUL_Clipped values indexed by time
3. Normalize labels using Scaler:
   - Fit scaler on training labels
   - Transform val/test/oot labels
4. Save TimeSeries objects and scaler

**Output:**
- Directory: `datamart/gold/label/lstm/n_cmapss/snapshot_date={YYYY-MM-DD}/`
- Format: Pickle (Darts TimeSeries)
- Files:
  - `train/targets.pkl` (list of 39 TimeSeries)
  - `val/targets.pkl` (list of 7 TimeSeries)
  - `test/targets.pkl` (list of 9 TimeSeries)
  - `oot/targets.pkl` (list of 19 TimeSeries)
  - `target_scaler.pkl` (fitted Scaler)
  - `units.pkl` (list of unit IDs)

**Task ID:** `gold_label_lstm_ncmapss`

---

### 5. gold_label_xgboost_ncmapss

**Purpose:** Prepare labels in XGBoost-compatible DataFrame format

**Script:** `scripts/gold_label_xgboost_ncmapss.py`

**Input:**
- Label base DataFrames from gold_label_base_ncmapss

**Processing:**
1. Load train/val/test/oot parquet files
2. For each split:
   - Extract label columns: `unit`, `time`, `RUL_Clipped`
   - Keep as DataFrame (no normalization needed for XGBoost)
3. Save label DataFrames

**Output:**
- Directory: `datamart/gold/label/xgboost/n_cmapss/snapshot_date={YYYY-MM-DD}/`
- Format: Parquet (DataFrame)
- Files:
  - `train/labels.parquet` (26.9M rows × 3 columns)
  - `val/labels.parquet` (6.9M rows × 3 columns)
  - `test/labels.parquet` (7.3M rows × 3 columns)
  - `oot/labels.parquet` (13.7M rows × 3 columns)

**Task ID:** `gold_label_xgboost_ncmapss`

---

### 6. gold_feature_lstm_ncmapss

**Purpose:** Create LSTM-compatible time series features

**Script:** `scripts/gold_feature_lstm_ncmapss.py`

**Input:**
- Label base DataFrames from gold_label_base_ncmapss

**Configuration:**
```python
SELECTED_FEATURES = [
    'HPC Outlet Pressure',
    'LPT Coolant Bleed',
    'Fan Inlet Pressure',
    'Demanded Fan Speed',
    'Fan Speed',
    'Core Speed',
    'Pressure in Bypass Duct',
    'Fuel Flow Ratio',
    'LPT Outlet Temperature',
    'Altitude',
    'Mach Number',
    'Throttle Resolver Angle'
]

SEQUENCE_LENGTH = 30  # Lookback window
```

**Processing:**
1. Load train/val/test/oot DataFrames
2. Select 12 features from SELECTED_FEATURES
3. For each split:
   - Group by engine unit
   - Create Darts TimeSeries objects (one per engine)
   - Each TimeSeries contains 12 feature values indexed by time
4. Normalize features using Scaler:
   - Fit scaler on training features
   - Transform val/test/oot features
5. Save TimeSeries objects and scaler

**Output:**
- Directory: `datamart/gold/feature/lstm/n_cmapss/snapshot_date={YYYY-MM-DD}/`
- Format: Pickle (Darts TimeSeries)
- Files:
  - `train/covariates.pkl` (list of 39 TimeSeries, each shape: (time_steps, 12))
  - `val/covariates.pkl` (list of 7 TimeSeries)
  - `test/covariates.pkl` (list of 9 TimeSeries)
  - `oot/covariates.pkl` (list of 19 TimeSeries)
  - `feature_scaler.pkl` (fitted Scaler)
  - `units.pkl` (list of unit IDs)

**Task ID:** `gold_feature_lstm_ncmapss`

---

### 7. gold_feature_xgboost_ncmapss

**Purpose:** Create time-windowed features for XGBoost

**Script:** `scripts/gold_feature_xgboost_ncmapss.py`

**Input:**
- Label base DataFrames from gold_label_base_ncmapss

**Configuration:**
```python
SELECTED_FEATURES = [
    # Same 12 features as LSTM
]

WINDOWS = {
    'short': 5,    # Last 5 cycles
    'medium': 15,  # Last 15 cycles
    'long': 30     # Last 30 cycles
}
```

**Processing:**
1. Load train/val/test/oot DataFrames
2. Convert features to float32 (memory optimization)
3. For each of 12 features, create:
   - Rolling mean (3 windows) → 36 features
   - Rolling std (3 windows) → 36 features
   - Acceleration (mean_short - mean_medium) → 12 features
4. Process features in batches with garbage collection
5. Save engineered DataFrames

**Output:**
- Directory: `datamart/gold/feature/xgboost/n_cmapss/snapshot_date={YYYY-MM-DD}/`
- Format: Parquet (DataFrame, float32)
- Files:
  - `train/features.parquet` (26.9M rows × 96 features)
  - `val/features.parquet` (6.9M rows × 96 features)
  - `test/features.parquet` (7.3M rows × 96 features)
  - `oot/features.parquet` (13.7M rows × 96 features)
  - `feature_columns.txt` (list of 96 feature names)

**Task ID:** `gold_feature_xgboost_ncmapss`

---

## Why Separate Label Stores?

### The Problem

While both LSTM and XGBoost models use the **same underlying labels** (RUL_Clipped values with identical train/val/test/oot splits), they require **different data formats**:

| Aspect | LSTM Requirements | XGBoost Requirements |
|--------|-------------------|----------------------|
| **Data structure** | 3D sequences (samples, timesteps, features) | 2D tabular (samples, features) |
| **Label format** | TimeSeries objects | DataFrame/arrays |
| **Temporal encoding** | Explicit time sequences | Flattened rows |
| **Normalization** | Normalized to [0, 1] | Raw values (tree-based models don't need normalization) |
| **Indexing** | Per-engine time series | Flat row index |

### The Solution

**Three-tier label architecture:**

1. **gold_label_base_ncmapss**: Creates the ground truth
   - RUL_Clipped calculation
   - Train/val/test/oot splits
   - Saved as DataFrame (universal format)

2. **gold_label_lstm_ncmapss**: LSTM-specific formatting
   - Converts DataFrame → TimeSeries objects
   - Normalizes to [0, 1]
   - Maintains temporal structure

3. **gold_label_xgboost_ncmapss**: XGBoost-specific formatting
   - Keeps as DataFrame
   - No normalization
   - Simplified structure (unit, time, RUL_Clipped)

### Benefits

1. **Separation of Concerns**
   - Data splitting logic separated from format conversion
   - Each model gets labels in optimal format
   - Changes to split logic don't affect model-specific formatting

2. **Modularity**
   - Can update LSTM label formatting without affecting XGBoost
   - Can add new models (e.g., Transformer) by creating new label formatter

3. **Clarity**
   - Pipeline explicitly shows format conversions
   - Easier to debug format-related issues
   - Clear lineage: base → model-specific

4. **Efficiency**
   - Base labels computed once
   - Parallel processing of LSTM and XGBoost branches
   - No redundant split calculations

## Directory Structure

```
datamart/
├── bronze/
│   └── n_cmapss/
│       └── snapshot_date=2023-01-01/
│           ├── W.parquet              # Scenario descriptors (4 features)
│           ├── X_s.parquet            # Physical sensors (14 features)
│           ├── X_v.parquet            # Virtual sensors (14 features)
│           ├── A.parquet              # Auxiliary data (5 features)
│           └── Y.parquet              # RUL target (1 feature)
│
├── silver/
│   └── n_cmapss/
│       └── snapshot_date=2023-01-01/
│           └── df_combined.parquet   # Cleaned data (38 columns)
│
└── gold/
    ├── label_base/                    # Base labels (DataFrame format)
    │   └── n_cmapss/
    │       └── snapshot_date=2023-01-01/
    │           ├── df_train.parquet   # 39 engines, 26.9M rows
    │           ├── df_val.parquet     # 7 engines, 6.9M rows
    │           ├── df_test.parquet    # 9 engines, 7.3M rows
    │           ├── df_oot.parquet     # 19 engines, 13.7M rows
    │           └── split_metadata.json
    │
    ├── label/
    │   ├── lstm/                      # LSTM labels (TimeSeries format)
    │   │   └── n_cmapss/
    │   │       └── snapshot_date=2023-01-01/
    │   │           ├── train/
    │   │           │   ├── targets.pkl      # List of 39 TimeSeries
    │   │           │   └── units.pkl
    │   │           ├── val/
    │   │           │   ├── targets.pkl      # List of 7 TimeSeries
    │   │           │   └── units.pkl
    │   │           ├── test/
    │   │           │   ├── targets.pkl      # List of 9 TimeSeries
    │   │           │   └── units.pkl
    │   │           ├── oot/
    │   │           │   ├── targets.pkl      # List of 19 TimeSeries
    │   │           │   └── units.pkl
    │   │           └── target_scaler.pkl
    │   │
    │   └── xgboost/                   # XGBoost labels (DataFrame format)
    │       └── n_cmapss/
    │           └── snapshot_date=2023-01-01/
    │               ├── train/
    │               │   └── labels.parquet   # 26.9M rows × 3 cols
    │               ├── val/
    │               │   └── labels.parquet   # 6.9M rows × 3 cols
    │               ├── test/
    │               │   └── labels.parquet   # 7.3M rows × 3 cols
    │               └── oot/
    │                   └── labels.parquet   # 13.7M rows × 3 cols
    │
    └── feature/
        ├── lstm/                      # LSTM features (TimeSeries format)
        │   └── n_cmapss/
        │       └── snapshot_date=2023-01-01/
        │           ├── train/
        │           │   ├── covariates.pkl   # List of 39 TimeSeries (×12 features)
        │           │   └── units.pkl
        │           ├── val/
        │           │   ├── covariates.pkl   # List of 7 TimeSeries
        │           │   └── units.pkl
        │           ├── test/
        │           │   ├── covariates.pkl   # List of 9 TimeSeries
        │           │   └── units.pkl
        │           ├── oot/
        │           │   ├── covariates.pkl   # List of 19 TimeSeries
        │           │   └── units.pkl
        │           └── feature_scaler.pkl
        │
        └── xgboost/                   # XGBoost features (DataFrame format)
            └── n_cmapss/
                └── snapshot_date=2023-01-01/
                    ├── train/
                    │   └── features.parquet # 26.9M rows × 96 features
                    ├── val/
                    │   └── features.parquet # 6.9M rows × 96 features
                    ├── test/
                    │   └── features.parquet # 7.3M rows × 96 features
                    ├── oot/
                    │   └── features.parquet # 13.7M rows × 96 features
                    └── feature_columns.txt
```

## Data Flow Details

### Bronze Layer

**Input:** 7 HDF5 files (54,874,178 time steps across 74 engines)

**Output:** Parquet files with raw arrays

**Key Transformations:**
- Load HDF5 arrays
- Concatenate dev/test splits
- Add dataset identifier
- Save as parquet

---

### Silver Layer

**Input:** Bronze parquet files

**Output:** Single cleaned DataFrame (54,874,178 rows × 38 columns)

**Key Transformations:**
- Rename columns (human-readable)
- Create composite unit IDs
- Data validation
- Save as parquet

---

### Gold Layer - Base Labels

**Input:** Silver DataFrame

**Output:** 4 split DataFrames (train/val/test/oot)

**Key Transformations:**
- Create RUL_Clipped (clip at 90 cycles)
- Split by engine units:
  - OOT: Datasets 2, 7 (19 engines)
  - Internal: Datasets 1, 3, 4, 5, 6 (55 engines)
    - Test: 15% of internal (9 engines)
    - Train+Val: 85% of internal (46 engines)
      - Val: 15% of train+val (7 engines)
      - Train: 85% of train+val (39 engines)

---

### Gold Layer - LSTM Branch

**Input:** Base label DataFrames

**LSTM Labels:**
- Convert to TimeSeries objects (per-engine sequences)
- Normalize to [0, 1]
- Save as pickle files

**LSTM Features:**
- Select 12 features
- Convert to TimeSeries objects (per-engine sequences)
- Normalize to [0, 1]
- Save as pickle files

**Final Output for LSTM Model:**
- Covariates: List[TimeSeries] with shape (time_steps, 12)
- Targets: List[TimeSeries] with shape (time_steps, 1)
- Both normalized to [0, 1]

---

### Gold Layer - XGBoost Branch

**Input:** Base label DataFrames

**XGBoost Labels:**
- Extract unit, time, RUL_Clipped
- Keep as DataFrame
- Save as parquet

**XGBoost Features:**
- Select 12 base features
- Create rolling statistics:
  - Mean (3 windows × 12 features = 36)
  - Std (3 windows × 12 features = 36)
  - Acceleration (12 features)
- Total: 96 engineered features
- Save as parquet (float32)

**Final Output for XGBoost Model:**
- Features: DataFrame (samples, 96 features)
- Labels: DataFrame (samples, 3 columns)
- Both in tabular format

---

## Implementation Files

### Scripts (Entry Points)

| File | Task ID | Layer | Purpose |
|------|---------|-------|---------|
| `bronze_ncmapss.py` | `bronze_ncmapss` | Bronze | Load raw HDF5 files |
| `silver_ncmapss.py` | `silver_ncmapss` | Silver | Clean & standardize |
| `gold_label_base_ncmapss.py` | `gold_label_base_ncmapss` | Gold | Create base labels + splits |
| `gold_label_lstm_ncmapss.py` | `gold_label_lstm_ncmapss` | Gold | Format labels for LSTM |
| `gold_label_xgboost_ncmapss.py` | `gold_label_xgboost_ncmapss` | Gold | Format labels for XGBoost |
| `gold_feature_lstm_ncmapss.py` | `gold_feature_lstm_ncmapss` | Gold | Create LSTM features |
| `gold_feature_xgboost_ncmapss.py` | `gold_feature_xgboost_ncmapss` | Gold | Create XGBoost features |

### Utils (Processing Logic)

| File | Purpose |
|------|---------|
| `utils/data_processing_bronze_ncmapss.py` | Bronze layer processing logic |
| `utils/data_processing_silver_ncmapss.py` | Silver layer processing logic |
| `utils/data_processing_gold_label_base.py` | Base label creation + splitting |
| `utils/data_processing_gold_label_lstm.py` | LSTM label formatting |
| `utils/data_processing_gold_label_xgboost.py` | XGBoost label formatting |
| `utils/data_processing_gold_feature_lstm.py` | LSTM feature engineering |
| `utils/data_processing_gold_feature_xgboost.py` | XGBoost feature engineering |

### DAG

| File | Purpose |
|------|---------|
| `dags/dag.py` | Airflow DAG definition with task dependencies |

## Task Dependencies in Airflow

```python
# Linear pipeline
start_pipeline >> bronze_ncmapss >> silver_ncmapss >> gold_label_base_ncmapss

# LSTM branch (sequential)
gold_label_base_ncmapss >> gold_label_lstm_ncmapss >> gold_feature_lstm_ncmapss

# XGBoost branch (sequential)
gold_label_base_ncmapss >> gold_label_xgboost_ncmapss >> gold_feature_xgboost_ncmapss

# Converge to end
[gold_feature_lstm_ncmapss, gold_feature_xgboost_ncmapss] >> end_pipeline
```

**Execution Order:**
1. Start → Bronze → Silver → Gold Label Base (linear, sequential)
2. Gold Label Base → [LSTM Label, XGBoost Label] (parallel)
3. LSTM Label → LSTM Feature (sequential)
4. XGBoost Label → XGBoost Feature (sequential)
5. [LSTM Feature, XGBoost Feature] → End (converge)

**Total Tasks:** 9 (start, 5 processing, 2 model-specific labels, 2 model-specific features, end)

## Related Documentation

- [Data Splitting Strategy](01-data-splitting-strategy.md) - Detailed explanation of train/val/test/oot splits
- [Medallion Architecture Pipeline](02-data-pipeline-medallion-architecture.md) - Bronze/Silver/Gold layer transformations
- Notebook: `N CMAPSS ML Model.ipynb` - Original analysis and feature engineering

## Summary

This pipeline architecture provides:

1. **Clear separation of concerns** - Each layer has a specific responsibility
2. **Model-agnostic base layers** - Bronze, silver, and base labels work for any model
3. **Model-specific gold layers** - LSTM and XGBoost get data in optimal formats
4. **Parallel processing** - LSTM and XGBoost branches run independently
5. **Reproducibility** - Snapshot dates enable time-travel and version control
6. **Scalability** - New models can be added by creating new gold layer branches

The design ensures that data transformations are explicit, testable, and maintainable while supporting the different requirements of sequence-based (LSTM) and tabular (XGBoost) models.
