# Data Pipeline: Medallion Architecture (Bronze → Silver → Gold)

This document describes the complete data engineering pipeline used in the N-CMAPSS ML Model notebook, following the medallion architecture pattern with bronze (raw), silver (cleaned), and gold (model-specific) layers.

## Table of Contents

- [Overview](#overview)
- [Bronze Layer: Raw Data](#bronze-layer-raw-data)
- [Silver Layer: Cleaned & Standardized Data](#silver-layer-cleaned--standardized-data)
- [Gold Layer: Model-Specific Features](#gold-layer-model-specific-features)
  - [Gold Layer for LSTM](#gold-layer-for-lstm)
  - [Gold Layer for XGBoost](#gold-layer-for-xgboost)
- [Summary Comparison](#summary-comparison)

## Overview

The data pipeline follows the **medallion architecture**:

```
Bronze (Raw) → Silver (Cleaned) → Gold (Model-Specific)
    ↓              ↓                    ↓
  HDF5        Standardized DF    LSTM Features + XGBoost Features
```

**Pipeline flow:**
1. **Bronze**: Load raw HDF5 files from 7 N-CMAPSS datasets
2. **Silver**: Clean, standardize, and combine into a single DataFrame
3. **Gold (LSTM)**: Select 12 features, create time series sequences
4. **Gold (XGBoost)**: Create 89 time-windowed engineered features

## Bronze Layer: Raw Data

### Description

The bronze layer consists of **raw HDF5 files** from the N-CMAPSS (NASA Commercial Modular Aero-Propulsion System Simulation) dataset containing turbofan engine degradation simulation data.

### Data Sources

**7 HDF5 datasets** from N-CMAPSS:
- `Data/N-CMAPSS_DS01.h5` (10 engines)
- `Data/N-CMAPSS_DS02.h5` (9 engines)
- `Data/N-CMAPSS_DS03.h5` (15 engines)
- `Data/N-CMAPSS_DS04.h5` (10 engines)
- `Data/N-CMAPSS_DS05.h5` (10 engines)
- `Data/N-CMAPSS_DS06.h5` (10 engines)
- `Data/N-CMAPSS_DS07.h5` (10 engines)

**Total: 74 engines, 54,874,178 time steps**

### Loading Function

**Function:** `load_data(filenames)`

```python
filenames = [
    'Data/N-CMAPSS_DS01.h5',
    'Data/N-CMAPSS_DS02.h5',
    'Data/N-CMAPSS_DS03.h5',
    'Data/N-CMAPSS_DS04.h5',
    'Data/N-CMAPSS_DS05.h5',
    'Data/N-CMAPSS_DS06.h5',
    'Data/N-CMAPSS_DS07.h5'
]

W, X_s, X_v, T, Y, A = load_data(filenames)
```

**Loading process:**
1. Opens each HDF5 file using `h5py.File()`
2. Reads both development (`_dev`) and test (`_test`) arrays
3. Concatenates dev and test data for each dataset
4. Concatenates all 7 datasets together
5. Adds dataset identifier to each row

### Data Structure

Each HDF5 file contains 6 array groups:

#### W_var (Scenario Descriptors - 4 features)
Operating conditions for each time step:
- `alt`: Altitude
- `Mach`: Mach Number
- `TRA`: Throttle Resolver Angle
- `T2`: Total Temperature at Fan Inlet

#### X_s_var (Physical Sensors - 14 features)
Direct sensor measurements:
- `T24`: LPC Outlet Temperature
- `T30`: HPC Inlet Temperature
- `T48`: HPT Outlet Temperature
- `T50`: LPT Outlet Temperature
- `P15`: Pressure in Bypass Duct
- `P2`: Fan Inlet Pressure
- `P21`: Engine Pressure Ratio
- `P24`: Corrected Fan Speed Ratio
- `Ps30`: HPC Outlet Static Pressure
- `P40`: Bypass Ratio
- `P50`: Total Pressure at LPT Outlet
- `Nf`: Fan Speed
- `Nc`: Core Speed
- `Wf`: Fuel Flow

#### X_v_var (Virtual Sensors - 14 features)
Computed/derived measurements:
- `T40`: Total Temperature at Burner Outlet
- `P30`: HPC Outlet Pressure
- `P45`: Total Pressure at HPT Outlet
- `W21`: Fan Flow
- `W22`: LPC Flow
- `W25`: HPC Flow
- `W31`: HPT Coolant Bleed
- `W32`: LPT Coolant Bleed
- `W48`: Bleed Enthalpy
- `W50`: Demanded Fan Speed
- `SmFan`: Fan Stall Margin
- `SmLPC`: LPC Stall Margin
- `SmHPC`: HPC Stall Margin
- `phi`: Fuel Flow Ratio

#### A_var (Auxiliary Data - 4 features)
Metadata for each time step:
- `unit`: Engine unit identifier (within dataset)
- `cycle`: Time cycle/step
- `Fc`: Flight Class
- `hs`: Health Status

#### Y_var (Target - 1 feature)
- Remaining Useful Life (RUL) in cycles

#### T_var (Degradation Parameters)
- Internal degradation parameters (not used in final models)

### Final Array Shapes

After loading and concatenating all datasets:
- `W`: (54,874,178, 4) - Scenario descriptors
- `X_s`: (54,874,178, 14) - Physical sensors
- `X_v`: (54,874,178, 14) - Virtual sensors
- `A`: (54,874,178, 5) - Auxiliary data (includes dataset ID)
- `Y`: (54,874,178, 1) - RUL target
- `T`: (54,874,178, n) - Degradation parameters (unused)

### Initial Processing

Minimal processing at bronze layer:
- Concatenation of dev and test splits
- Concatenation of all 7 datasets
- Addition of dataset number as first column in auxiliary data
- **No cleaning, transformation, or feature engineering**

## Silver Layer: Cleaned & Standardized Data

### Description

The silver layer transforms raw numpy arrays into a **cleaned, standardized pandas DataFrame** with human-readable column names and proper data types.

### Transformation Function

**Function:** `create_df(A_data, W_data, X_s_data, X_v_data, T_data, Y_data)`

```python
df_combined = create_df(A, W, X_s, X_v, T, Y)
```

### Transformations Applied

#### 1. Column Renaming

Applied `COLUMN_RENAME_MAP` to make column names descriptive:

```python
COLUMN_RENAME_MAP = {
    'T24': 'LPC Outlet Temperature',
    'T30': 'HPC Inlet Temperature',
    'T48': 'HPT Outlet Temperature',
    'T50': 'LPT Outlet Temperature',
    'P15': 'Pressure in Bypass Duct',
    'P2': 'Fan Inlet Pressure',
    'Nf': 'Fan Speed',
    'Nc': 'Core Speed',
    'Wf': 'Fuel Flow',
    'P30': 'HPC Outlet Pressure',
    'W32': 'LPT Coolant Bleed',
    'W50': 'Demanded Fan Speed',
    'phi': 'Fuel Flow Ratio',
    'alt': 'Altitude',
    'Mach': 'Mach Number',
    'TRA': 'Throttle Resolver Angle',
    'T2': 'Total Temperature at Fan Inlet',
    # ... (33 total mappings)
}
```

#### 2. Unit Identifier Creation

Created composite unit identifier to ensure uniqueness across datasets:

```python
unit = f"DS{dataset:02d}_{unit_orig:03d}"
```

**Examples:**
- Dataset 1, Unit 1 → `DS01_001`
- Dataset 2, Unit 15 → `DS02_015`

#### 3. Data Integrity Checks

**Checks performed:**
- Missing values (NaN) detection - **Result: No NaNs found**
- Constant/non-trending feature identification
- Low variance feature analysis

#### 4. Data Validation

No data cleaning or imputation needed as the simulated data has:
- No missing values
- No outliers requiring treatment
- No duplicate rows
- Consistent data types

### Features Added/Modified

**New columns:**
- `unit`: Composite identifier (format: `DS{dataset:02d}_{unit_orig:03d}`)
- `Remaining Useful Life`: Target variable from Y_data

**Preserved and renamed columns:**
- `dataset`: Dataset number (1-7)
- `unit_orig`: Original unit number within dataset
- `time`: Cycle/time step (from `cycle` in A_data)
- `Fc` → `Flight Class`
- `hs` → `Health Status`
- All 32 sensor and operational condition features (with human-readable names)

### Silver Layer Schema

**DataFrame:** `df_combined`

**Shape:** (54,874,178 rows, 38 columns)

**Column categories:**

1. **Metadata (4 columns):**
   - `dataset`: int (1-7)
   - `unit`: string (e.g., "DS01_001")
   - `unit_orig`: int
   - `time`: int (cycle number)

2. **Operating Conditions (4 columns):**
   - Altitude
   - Mach Number
   - Throttle Resolver Angle
   - Total Temperature at Fan Inlet

3. **Physical Sensors (14 columns):**
   - Temperature measurements (4)
   - Pressure measurements (6)
   - Speed measurements (2)
   - Flow measurements (2)

4. **Virtual Sensors (14 columns):**
   - Computed temperatures (1)
   - Computed pressures (2)
   - Computed flows (5)
   - Stall margins (3)
   - Other derived metrics (3)

5. **Auxiliary (2 columns):**
   - Flight Class
   - Health Status

6. **Target (1 column):**
   - Remaining Useful Life (0-125 cycles)

### Storage

**File:** `data/df_combined.pkl`
**Format:** Pickled pandas DataFrame
**Size:** ~54.9M rows × 38 columns

## Gold Layer: Model-Specific Features

The gold layer splits into two model-specific feature sets: one optimized for LSTM (sequence-based) and one for XGBoost (tabular with engineered features).

## Gold Layer for LSTM

### Description

The LSTM gold layer creates **time series sequences** from selected features, optimized for recurrent neural network processing.

### Transformation Process

#### 1. Feature Selection

Selected **12 features** based on correlation analysis with RUL:

```python
selected_features = [
    'HPC Outlet Pressure',          # Virtual sensor
    'LPT Coolant Bleed',           # Virtual sensor
    'Fan Inlet Pressure',          # Physical sensor
    'Demanded Fan Speed',          # Virtual sensor
    'Fan Speed',                   # Physical sensor
    'Core Speed',                  # Physical sensor
    'Pressure in Bypass Duct',     # Physical sensor
    'Fuel Flow Ratio',             # Virtual sensor
    'LPT Outlet Temperature',      # Physical sensor
    'Altitude',                    # Operating condition
    'Mach Number',                 # Operating condition
    'Throttle Resolver Angle'      # Operating condition
]

# Keep required auxiliary columns
required_aux_cols = ['unit', 'time', 'Remaining Useful Life', 'dataset']

# Create gold DataFrame
gold_df = df_combined[required_aux_cols + final_features].copy()
```

#### 2. Target Engineering

**RUL Clipping:**

```python
RUL_CLIP_MAX = 90

gold_df['RUL_Clipped'] = gold_df['Remaining Useful Life'].clip(upper=RUL_CLIP_MAX)
```

**Rationale:**
- Original RUL range: 0-125 cycles
- Clipped RUL range: 0-90 cycles
- Prevents model from focusing too much on early-life predictions
- Focuses learning on critical degradation period

#### 3. Data Splitting

Split data by engine units into 4 sets:

```python
df_train, df_val, df_test, df_oot = split_data_by_unit(gold_df, TRAIN_SETS, OOT_SETS)
```

**Split distribution:**
- **Training**: 26,898,226 rows (39 engines) - Datasets 1, 3, 4, 5, 6
- **Validation**: 6,905,795 rows (7 engines) - Internal split
- **Test**: 7,333,005 rows (9 engines) - Internal split
- **OOT**: 13,737,152 rows (19 engines) - Datasets 2, 7

See [data-splitting-strategy.md](data-splitting-strategy.md) for details.

#### 4. Time Series Creation

**Function:** `create_darts_series_from_df(df, features, target='RUL_Clipped')`

Converts DataFrame to **Darts TimeSeries objects** (one per engine):

```python
train_covariates, train_targets, train_units = create_darts_series_from_df(
    df_train,
    final_features
)
```

**Output structure:**
- `covariates_list`: List of TimeSeries, each shape (time_steps, 12)
- `targets_list`: List of TimeSeries, each shape (time_steps, 1)
- `units_list`: List of engine unit identifiers

**Example for one engine:**
- Engine DS01_001 with 1000 cycles
- Covariates: TimeSeries (1000, 12)
- Target: TimeSeries (1000, 1)

#### 5. Normalization

**Feature normalization:**

```python
from darts.dataprocessing.transformers import Scaler

# Fit on training data
feature_scaler = Scaler()
train_covariates_scaled = feature_scaler.fit_transform(train_covariates)

# Apply to validation/test
val_covariates_scaled = feature_scaler.transform(val_covariates)
test_covariates_scaled = feature_scaler.transform(test_covariates)
```

**Target normalization:**

```python
target_scaler = Scaler()
train_targets_scaled = target_scaler.fit_transform(train_targets)
val_targets_scaled = target_scaler.transform(val_targets)
test_targets_scaled = target_scaler.transform(test_targets)
```

**Scaling method:** Min-max scaling to [0, 1] range

### LSTM Model Configuration

**Sequence parameters:**
```python
SEQUENCE_LENGTH = 30  # Lookback window: 30 time steps
```

**Model architecture:**
```python
HIDDEN_DIM = 32       # Hidden layer size
N_RNN_LAYERS = 2      # Number of LSTM layers
```

**Input/Output shapes:**
- Input: (batch_size, 30, 12) - 30 time steps × 12 features
- Output: (batch_size, 1) - RUL prediction

### LSTM Gold Layer Schema

**DataFrames:**
- `gold_df`: (54,874,178, 17) - Full dataset
- `df_train`: (26,898,226, 17) - Training set
- `df_val`: (6,905,795, 17) - Validation set
- `df_test`: (7,333,005, 17) - Test set
- `df_oot`: (13,737,152, 17) - OOT set

**Columns (17 total):**
- Metadata: `unit`, `time`, `dataset` (3)
- Features: 12 selected sensor/operational features
- Targets: `Remaining Useful Life`, `RUL_Clipped` (2)

### Key Functions

1. `split_data_by_unit(df, train_sets, oot_sets)`: Splits data by units
2. `create_darts_series_from_df(df, features, target)`: Creates TimeSeries objects
3. `Scaler()`: Darts transformer for normalization
4. `fit_transform()` / `transform()`: Apply scaling

## Gold Layer for XGBoost

### Description

The XGBoost gold layer creates **time-windowed engineered features** to capture temporal patterns explicitly, since XGBoost cannot process sequences directly.

### Transformation Process

#### 1. Multi-Scale Feature Engineering

**Function:** `create_multiscale_features(df, features)`

Creates rolling window statistics at **3 different time scales**:

```python
windows = {
    'short': 5,    # Last 5 cycles - captures immediate trends
    'medium': 15,  # Last 15 cycles - captures medium-term trends
    'long': 30     # Last 30 cycles - matches LSTM sequence length
}
```

#### 2. Features Created for Each Sensor

For each of the **12 selected features**, create:

**a) Rolling Mean (3 windows × 12 sensors = 36 features)**

```python
# Example for Fan Speed
df['Fan Speed_mean_short'] = df.groupby('unit')['Fan Speed'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)
df['Fan Speed_mean_medium'] = df.groupby('unit')['Fan Speed'].transform(
    lambda x: x.rolling(window=15, min_periods=1).mean()
)
df['Fan Speed_mean_long'] = df.groupby('unit')['Fan Speed'].transform(
    lambda x: x.rolling(window=30, min_periods=1).mean()
)
```

**b) Rolling Standard Deviation (3 windows × 12 sensors = 36 features)**

```python
# Example for Core Speed
df['Core Speed_std_short'] = df.groupby('unit')['Core Speed'].transform(
    lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
)
df['Core Speed_std_medium'] = df.groupby('unit')['Core Speed'].transform(
    lambda x: x.rolling(window=15, min_periods=1).std().fillna(0)
)
df['Core Speed_std_long'] = df.groupby('unit')['Core Speed'].transform(
    lambda x: x.rolling(window=30, min_periods=1).std().fillna(0)
)
```

**c) Acceleration Features (12 sensors × 1 = 12 features)**

Captures rate of change by comparing window means:

```python
# Example for Altitude
df['Altitude_acceleration'] = (
    df['Altitude_mean_short'] - df['Altitude_mean_medium']
)
```

**Formula:** `acceleration = mean_short - mean_medium`

**Interpretation:**
- Positive: Feature increasing (accelerating)
- Negative: Feature decreasing (decelerating)
- Near zero: Feature stable

#### 3. Total Engineered Features

**Feature breakdown:**
- Original features: 12
- Rolling means: 36 (12 × 3 windows)
- Rolling stds: 36 (12 × 3 windows)
- Acceleration: 12 (12 × 1)
- **Total engineered: 96 features**
- Plus 5 auxiliary columns: `unit`, `time`, `dataset`, `Remaining Useful Life`, `RUL_Clipped`
- **Grand total: 101 columns**

**Features used for training:** 89 (excludes unit, time, dataset, Remaining Useful Life, RUL_Clipped)

### Memory Optimizations

The feature engineering process handles large data efficiently:

**1. Float32 Conversion**

```python
df[features] = df[features].astype('float32')
```
- Reduces memory usage by 50% (vs float64)
- Sufficient precision for sensor data

**2. Batch Processing**

```python
# Process 3 sensors at a time
batch_size = 3
for i in range(0, len(features), batch_size):
    batch = features[i:i + batch_size]
    # Create rolling features for batch
    # ...
    gc.collect()  # Free memory after each batch
```

**3. Efficient Storage**

```python
# Save engineered features
df_train_xgb.to_pickle('data/df_train_xgb.pkl')
df_val_xgb.to_pickle('data/df_val_xgb.pkl')
df_test_xgb.to_pickle('data/df_test_xgb.pkl')
df_oot_xgb.to_pickle('data/df_oot_xgb.pkl')
```

### XGBoost Gold Layer Schema

**DataFrames:**
- `df_train_xgb`: (26,898,226, 101)
- `df_val_xgb`: (6,905,795, 101)
- `df_test_xgb`: (7,333,005, 101)
- `df_oot_xgb`: (13,737,152, 101)

**Column structure (101 total):**

1. **Metadata (5 columns):**
   - `unit`, `time`, `dataset`, `Remaining Useful Life`, `RUL_Clipped`

2. **Original Features (12 columns):**
   - HPC Outlet Pressure
   - LPT Coolant Bleed
   - Fan Inlet Pressure
   - Demanded Fan Speed
   - Fan Speed
   - Core Speed
   - Pressure in Bypass Duct
   - Fuel Flow Ratio
   - LPT Outlet Temperature
   - Altitude
   - Mach Number
   - Throttle Resolver Angle

3. **Rolling Mean Features (36 columns):**
   - `{feature}_mean_short` (12)
   - `{feature}_mean_medium` (12)
   - `{feature}_mean_long` (12)

4. **Rolling Std Features (36 columns):**
   - `{feature}_std_short` (12)
   - `{feature}_std_medium` (12)
   - `{feature}_std_long` (12)

5. **Acceleration Features (12 columns):**
   - `{feature}_acceleration` (12)

**Training features:** Stored in `xgb_feature_cols` list (89 features)

### XGBoost Model Configuration

**Training parameters:**
```python
xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
```

**Input/Output:**
- Input: (batch_size, 89) - 89 engineered features
- Output: (batch_size, 1) - RUL prediction

### Key Implementation Functions

**`create_multiscale_features(df, features)`**

Main feature engineering function:

```python
def create_multiscale_features(df, features):
    """
    Create multi-scale time-windowed features for XGBoost.

    For each feature:
    1. Calculate rolling mean for short/medium/long windows
    2. Calculate rolling std for short/medium/long windows
    3. Calculate acceleration (mean_short - mean_medium)

    Args:
        df: DataFrame with time series data
        features: List of feature names to engineer

    Returns:
        DataFrame with original + engineered features
    """
    # Convert to float32 for memory efficiency
    df[features] = df[features].astype('float32')

    # Define time windows
    windows = {'short': 5, 'medium': 15, 'long': 30}

    # Process features in batches
    batch_size = 3
    for i in range(0, len(features), batch_size):
        batch = features[i:i + batch_size]

        # Create rolling statistics
        for feature in batch:
            for window_name, window_size in windows.items():
                # Rolling mean
                df[f'{feature}_mean_{window_name}'] = df.groupby('unit')[feature].transform(
                    lambda x: x.rolling(window=window_size, min_periods=1).mean()
                )

                # Rolling std
                df[f'{feature}_std_{window_name}'] = df.groupby('unit')[feature].transform(
                    lambda x: x.rolling(window=window_size, min_periods=1).std().fillna(0)
                )

            # Acceleration: difference between short and medium window means
            df[f'{feature}_acceleration'] = (
                df[f'{feature}_mean_short'] - df[f'{feature}_mean_medium']
            )

        gc.collect()  # Free memory

    return df
```

## Comparison: LSTM vs XGBoost Gold Layers

| Aspect | LSTM Gold Layer | XGBoost Gold Layer |
|--------|----------------|-------------------|
| **Purpose** | Sequential pattern learning | Explicit temporal feature engineering |
| **Data structure** | 3D sequences (samples, timesteps, features) | 2D tabular (samples, features) |
| **Feature count** | 12 raw features | 12 raw + 84 engineered = 96 features |
| **Temporal encoding** | Implicit (LSTM learns from sequences) | Explicit (rolling windows) |
| **Sequence length** | 30 time steps lookback | N/A (encoded in rolling windows) |
| **Memory format** | Darts TimeSeries objects | Pandas DataFrame (float32) |
| **Normalization** | Min-max scaling [0, 1] | None (XGBoost handles raw values) |
| **Target** | RUL_Clipped (scaled) | RUL_Clipped (raw) |
| **Input shape** | (batch, 30, 12) | (batch, 89) |
| **Processing** | Per-engine time series | Flattened with grouped rolling ops |
| **Model type** | Recurrent Neural Network | Gradient Boosted Trees |

### When to Use Each Approach

**LSTM Gold Layer (Sequence-based):**
- When temporal dependencies are complex and non-linear
- When you want the model to learn temporal patterns automatically
- When you have sufficient data for deep learning
- When interpretability is less critical

**XGBoost Gold Layer (Engineered features):**
- When you want explicit control over temporal features
- When interpretability is important (feature importance)
- When training time is a concern (faster than LSTM)
- When you have domain knowledge to guide feature engineering

## Summary Comparison

| Layer | Description | Shape | Key Transformations | Storage |
|-------|-------------|-------|---------------------|---------|
| **Bronze** | Raw HDF5 arrays | (54.9M, 32) | Load + concatenate 7 datasets | In-memory arrays |
| **Silver** | Cleaned DataFrame | (54.9M, 38) | Rename columns, create unit IDs, validate | `df_combined.pkl` |
| **Gold (LSTM)** | Time series sequences | 74 engines × variable time steps × 12 features | Feature selection, RUL clipping, normalization, TimeSeries creation | Darts TimeSeries objects |
| **Gold (XGBoost)** | Time-windowed tabular | (54.9M, 101) | Rolling statistics (mean, std), acceleration features, float32 conversion | `df_*_xgb.pkl` files |

### Data Flow Summary

```
Bronze Layer (Raw HDF5)
└─> 7 datasets × (dev + test) splits
    └─> Concatenate to numpy arrays (W, X_s, X_v, A, Y, T)
        └─> Shape: (54,874,178, 32 features)

Silver Layer (Cleaned DataFrame)
└─> Convert arrays to DataFrame
    └─> Rename columns (human-readable)
        └─> Create composite unit IDs
            └─> Validate data integrity
                └─> Shape: (54,874,178, 38 columns)
                    └─> Save: df_combined.pkl

Gold Layer (Model-Specific)
├─> LSTM Branch
│   └─> Select 12 features
│       └─> Clip RUL to 90
│           └─> Split by units (train/val/test/oot)
│               └─> Create TimeSeries objects
│                   └─> Normalize to [0,1]
│                       └─> 74 engines × (timesteps, 12)
│
└─> XGBoost Branch
    └─> Select same 12 features
        └─> Clip RUL to 90
            └─> Split by units (train/val/test/oot)
                └─> Create multi-scale rolling features
                    └─> 36 means + 36 stds + 12 acceleration
                        └─> Shape: (54,874,178, 101)
                            └─> Save: df_*_xgb.pkl
```

## References

- Notebook file: `notebooks/N CMAPSS ML Model.ipynb`
- Bronze to Silver: Lines ~1-500 (data loading and DataFrame creation)
- Silver to Gold (LSTM): Lines ~1500-2000 (feature selection, splitting, TimeSeries)
- Silver to Gold (XGBoost): Lines ~2500-3000 (multi-scale feature engineering)
- Related documentation: [data-splitting-strategy.md](data-splitting-strategy.md)
