# Aircraft Engine RUL Prediction - System Architecture

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Architecture](#data-architecture)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Model Architecture](#model-architecture)
5. [Infrastructure Architecture](#infrastructure-architecture)
6. [Directory Structure](#directory-structure)
7. [Data Flow](#data-flow)
8. [Component Specifications](#component-specifications)

---

## System Overview

### Purpose

Production-grade MLOps system for predicting Remaining Useful Life (RUL) of aircraft engines using the NASA N-CMAPSS dataset.

### Key Components

- **Data Ingestion**: H5 to Parquet conversion (Bronze layer)
- **Data Processing**: Cleaning and standardization (Silver layer)
- **Feature Engineering**: Model-specific transformations (Gold layer)
- **ML Models**: LSTM (Darts) and XGBoost
- **Orchestration**: Apache Airflow with Docker
- **Monitoring**: Performance tracking and drift detection

### Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Orchestration | Apache Airflow | 2.11.0 |
| Containerization | Docker | 20.10+ |
| Data Processing | PySpark | 3.x |
| Deep Learning | PyTorch Lightning | 2.0+ |
| LSTM Framework | Darts | 0.27+ |
| Gradient Boosting | XGBoost | 1.7+ |
| Data Format | Parquet | PyArrow |

---

## Data Architecture

### Medallion Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAW DATA (H5)                            │
│                   scripts/data/*.h5                              │
│              ~350MB per monthly snapshot                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BRONZE LAYER                                │
│               datamart/bronze/n_cmapss/                          │
│     - Raw data in Parquet format                                 │
│     - Preserves all original features                            │
│     - Partitioned by snapshot_date                               │
│     - Size: ~1.75 GB per snapshot                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SILVER LAYER                                │
│               datamart/silver/n_cmapss/                          │
│     - Cleaned and normalized data                                │
│     - Missing values handled                                     │
│     - Data quality checks applied                                │
│     - Size: ~1.78 GB per snapshot                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GOLD LAYER                                 │
│               datamart/gold/                                     │
│     ┌──────────────────────────────────────────────┐            │
│     │         Label Base (Shared)                  │            │
│     │   - RUL with 90-cycle clipping               │            │
│     │   - 12 renamed sensor columns                │            │
│     └──────────────────────────────────────────────┘            │
│              │                      │                            │
│              ▼                      ▼                            │
│     ┌──────────────────┐   ┌──────────────────┐                │
│     │  LSTM Features   │   │ XGBoost Features │                │
│     │  12 sensors      │   │ 96 features      │                │
│     │  TimeSeries fmt  │   │ Rolling windows  │                │
│     └──────────────────┘   └──────────────────┘                │
│              │                      │                            │
│              ▼                      ▼                            │
│     ┌──────────────────┐   ┌──────────────────┐                │
│     │  LSTM Labels     │   │ XGBoost Labels   │                │
│     │  TimeSeries fmt  │   │ DataFrame fmt    │                │
│     └──────────────────┘   └──────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE LAYER                              │
│              datamart/inference/                                 │
│     - Model predictions (RUL values)                             │
│     - Separated by model type (lstm/xgboost)                     │
│     - Includes actual vs predicted                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING LAYER                              │
│             datamart/monitoring/                                 │
│     - Performance metrics (RMSE, MAE)                            │
│     - Drift detection reports                                    │
│     - HTML/PNG visualizations                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Data Schema

#### Bronze Layer

```
├── unit: int (engine identifier)
├── cycle: int (operational cycle)
├── W: float (operational setting)
├── X_s: array[14] (auxiliary data)
├── X_v: array[14] (virtual sensors)
├── T: float (flight condition)
├── Y: float (RUL - Remaining Useful Life)
├── A: array[4] (auxiliary data)
└── snapshot_date: date
```

#### Silver Layer

```
├── dataset: string (DS01-DS08)
├── unit: string (formatted: DS01_001)
├── cycle: int
├── T: float (normalized)
├── W: float (normalized)
├── X_s_1 to X_s_14: float (normalized)
├── X_v_1 to X_v_14: float (normalized)
├── A_1 to A_4: float
├── Y (RUL): float
└── snapshot_date: date
```

#### Gold Layer (Label Base)

```
├── dataset: string
├── unit: string
├── cycle: int
├── Altitude: float
├── MachNumber: float
├── TRA: float
├── T2: float
├── ... (12 renamed sensor columns)
├── hs (RUL): float (clipped at 90)
└── snapshot_date: date
```

---

## Pipeline Architecture

### 4-Stage DAG Structure

```
┌──────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: INGESTION                           │
│                        (Monthly: 00:00)                              │
│                                                                       │
│   ┌──────────────┐                                                  │
│   │   H5 Files   │                                                  │
│   │  (Raw Data)  │                                                  │
│   └──────┬───────┘                                                  │
│          │                                                           │
│          ▼                                                           │
│   ┌──────────────┐                                                  │
│   │   Bronze     │ ──► Parquet format                              │
│   │  Ingestion   │     Preserve sequences                           │
│   └──────────────┘     ~1.75GB output                              │
│          │                                                           │
└──────────┼───────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      STAGE 2: PREPROCESSING                          │
│                        (Monthly: 01:00)                              │
│                                                                       │
│   ┌──────────────┐                                                  │
│   │   Silver     │ ──► Data cleaning                               │
│   │ Transformation│     Normalization                               │
│   └──────┬───────┘     Quality checks                              │
│          │                                                           │
│          ▼                                                           │
│   ┌──────────────┐                                                  │
│   │ Gold Label   │ ──► RUL clipping (90 cycles)                    │
│   │    Base      │     Column renaming                              │
│   └──────┬───────┘                                                  │
│          │                                                           │
│          ├─────────────┬─────────────┐                              │
│          ▼             ▼             ▼                              │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                        │
│   │  LSTM    │  │ XGBoost  │  │  LSTM    │                        │
│   │  Labels  │  │  Labels  │  │ Features │                        │
│   └──────────┘  └──────────┘  └──────────┘                        │
│        │             │             │                                 │
│        │             │             ▼                                 │
│        │             │      ┌──────────────┐                        │
│        │             │      │  XGBoost     │                        │
│        │             │      │  Features    │                        │
│        │             │      │ (96 features)│                        │
│        │             │      └──────────────┘                        │
└────────┼─────────────┼─────────┼───────────────────────────────────┘
         │             │         │
         ▼             ▼         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   STAGE 3: TRAINING & INFERENCE                      │
│                        (Monthly: 02:00)                              │
│                                                                       │
│   ┌──────────────┐                                                  │
│   │ Check Model  │ ──► Model exists?                               │
│   │   Exists?    │     Yes: Skip training                           │
│   └──────┬───────┘     No: Train (manual)                          │
│          │                                                           │
│          ▼                                                           │
│   ┌──────────────┐                                                  │
│   │  Inference   │ ──► Load model from model_bank/                 │
│   │   (LSTM or   │     Load features & scalers                      │
│   │   XGBoost)   │     Generate predictions                         │
│   └──────┬───────┘     Save to inference/                          │
│          │                                                           │
└──────────┼───────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       STAGE 4: MONITORING                            │
│                        (Monthly: 03:00)                              │
│                                                                       │
│   ┌──────────────┐                                                  │
│   │  Compute     │ ──► RMSE, MAE metrics                           │
│   │  Metrics     │     Per-engine performance                       │
│   └──────┬───────┘                                                  │
│          │                                                           │
│          ▼                                                           │
│   ┌──────────────┐                                                  │
│   │   Drift      │ ──► Statistical tests                           │
│   │  Detection   │     Threshold alerts                             │
│   └──────┬───────┘                                                  │
│          │                                                           │
│          ▼                                                           │
│   ┌──────────────┐                                                  │
│   │  Generate    │ ──► HTML reports                                │
│   │   Reports    │     PNG charts                                  │
│   └──────────────┘     JSON metrics                                │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Parallel Pipelines

The system runs **two independent pipelines** for LSTM and XGBoost:

#### LSTM Pipeline

```
lstm_01_ingestion
    ↓
lstm_02_preprocessing
    ↓
lstm_03_training_inference
    ↓
lstm_04_monitoring
```

#### XGBoost Pipeline

```
xgboost_01_ingestion
    ↓
xgboost_02_preprocessing
    ↓
xgboost_03_training_inference
    ↓
xgboost_04_monitoring
```

**Note**: Bronze layer is shared between pipelines (both can write/read independently).

---

## Model Architecture

### LSTM Model (Darts BlockRNNModel)

```
Input Features (12 sensors):
  - Altitude, MachNumber, TRA, T2, T24, T30
  - T50, P15, P2, P21, P24, Ps30

Architecture:
  ┌─────────────────────────────────┐
  │   Input: [batch, 30, 12]        │  ← 30-cycle sequence
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │   LSTM Layer 1 (50 units)       │
  │   - Hidden state: 50            │
  │   - Dropout: 0.2                │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │   LSTM Layer 2 (50 units)       │
  │   - Hidden state: 50            │
  │   - Dropout: 0.2                │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │   Dense Layer (1 unit)          │
  │   - Output: RUL prediction      │
  └─────────────────────────────────┘

Hyperparameters:
  - Input chunk length: 30 cycles
  - Hidden dimension: 50
  - Number of RNN layers: 2
  - Dropout: 0.2
  - Batch size: 256
  - Learning rate: 0.001
  - Max epochs: 50
  - Early stopping: 10 epochs
```

### XGBoost Model

```
Input Features (96 total):
  Base Features (12): Same 12 sensors

  Rolling Window Features (84):
    For each base feature:
      - 5-cycle window: mean, std, acceleration
      - 15-cycle window: mean, std, acceleration
      - 30-cycle window: mean, std, acceleration

    Total: 12 features × 7 statistics = 84 features

Architecture:
  ┌─────────────────────────────────┐
  │   Input: [n_samples, 96]        │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │   XGBoost Gradient Boosting     │
  │   - Max depth: 6                │
  │   - Learning rate: 0.05         │
  │   - N estimators: 500           │
  │   - Subsample: 0.8              │
  │   - Colsample bytree: 0.8       │
  │   - Objective: reg:squarederror │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │   Output: RUL prediction        │
  └─────────────────────────────────┘

Hyperparameters:
  - max_depth: 6
  - learning_rate: 0.05
  - n_estimators: 500
  - early_stopping_rounds: 20
  - subsample: 0.8
  - colsample_bytree: 0.8
```

### Model Comparison

| Aspect | LSTM | XGBoost |
|--------|------|---------|
| **Input** | Sequences (30 cycles) | Flattened features |
| **Features** | 12 raw sensors | 96 engineered features |
| **Training Time** | 2-4 hours (GPU) | 15-30 min (CPU) |
| **Inference Speed** | Moderate (~100 samples/sec) | Fast (~1000 samples/sec) |
| **Interpretability** | Low (black box) | High (feature importance) |
| **Typical RMSE** | 0.10-0.15 | 0.08-0.15 |
| **Memory Usage** | High (sequences) | Moderate (features) |
| **Best For** | Long-term dependencies | Tabular feature patterns |

---

## Infrastructure Architecture

### Docker Compose Services

```
┌──────────────────────────────────────────────────────────────────┐
│                     DOCKER HOST                                  │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              AIRFLOW WEBSERVER                            │  │
│  │  - Port: 8080 (host) → 8080 (container)                  │  │
│  │  - UI for DAG monitoring                                  │  │
│  │  - User authentication                                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            ▲                                      │
│                            │                                      │
│  ┌────────────────────────┴──────────────────────────────────┐  │
│  │              AIRFLOW SCHEDULER                            │  │
│  │  - DAG parsing and execution                              │  │
│  │  - Task scheduling                                        │  │
│  │  - Volume mounts:                                         │  │
│  │    • ./dags → /opt/airflow/dags                          │  │
│  │    • ./scripts → /opt/airflow/scripts                    │  │
│  │    • ./logs → /opt/airflow/logs                          │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│                            │                                      │
│                            ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │         POSTGRESQL DATABASE (Airflow Metadata)            │  │
│  │  - Stores DAG states                                      │  │
│  │  - Task execution history                                 │  │
│  │  - Connection configs                                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              REDIS (Task Queue)                           │  │
│  │  - Optional message broker                                │  │
│  │  - Task distribution                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │         SHARED VOLUME MOUNTS                              │  │
│  │  - scripts/ (code execution)                              │  │
│  │  - dags/ (workflow definitions)                           │  │
│  │  - logs/ (execution logs)                                 │  │
│  │  - datamart/ (persistent data storage)                    │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Volume Mounts

```
Host                          Container
────────────────────          ─────────────────────────────
./dags/                  →    /opt/airflow/dags/
./scripts/               →    /opt/airflow/scripts/
./logs/                  →    /opt/airflow/logs/
./plugins/               →    /opt/airflow/plugins/

Persistent Data Directories:
./scripts/datamart/bronze/    →  Shared across containers
./scripts/datamart/silver/    →  Shared across containers
./scripts/datamart/gold/      →  Shared across containers
./scripts/model_bank/         →  Shared across containers
```

---

## Directory Structure

```
MLE Proj LSTM/
│
├── dags/                              # Airflow DAG definitions
│   ├── dag_lstm_ingestion.py         # LSTM Stage 1
│   ├── dag_lstm_preprocessing.py     # LSTM Stage 2
│   ├── dag_lstm_training_inference.py # LSTM Stage 3
│   ├── dag_lstm_monitoring.py        # LSTM Stage 4
│   ├── dag_xgboost_ingestion.py      # XGBoost Stage 1
│   ├── dag_xgboost_preprocessing.py  # XGBoost Stage 2
│   ├── dag_xgboost_training_inference.py # XGBoost Stage 3
│   └── dag_xgboost_monitoring.py     # XGBoost Stage 4
│
├── scripts/                           # Python execution scripts
│   ├── data/                          # Raw H5 files
│   │   ├── N-CMAPSS_DS_2025-01-01.h5
│   │   ├── N-CMAPSS_DS_2025-02-01.h5
│   │   └── ... (monthly snapshots)
│   │
│   ├── model_bank/                    # Trained models & scalers
│   │   ├── darts_lstm_model.pkl      # LSTM model
│   │   ├── darts_lstm_model.pkl.ckpt # LSTM checkpoint
│   │   ├── darts_target_scaler.pkl   # LSTM target scaler
│   │   ├── darts_covariate_scaler.pkl # LSTM covariate scaler
│   │   └── xgboost_rul_model.pkl     # XGBoost model
│   │
│   ├── datamart/                      # Data lake (Medallion)
│   │   ├── bronze/                    # Stage 1 output
│   │   ├── silver/                    # Stage 2 output
│   │   ├── gold/                      # Stage 2 output
│   │   │   ├── label_base/           # Shared labels
│   │   │   ├── label/lstm/           # LSTM labels
│   │   │   ├── label/xgboost/        # XGBoost labels
│   │   │   ├── feature/lstm/         # LSTM features
│   │   │   └── feature/xgboost/      # XGBoost features
│   │   ├── inference/                # Stage 3 output
│   │   │   ├── lstm/                 # LSTM predictions
│   │   │   └── xgboost/              # XGBoost predictions
│   │   └── monitoring/               # Stage 4 output
│   │       ├── lstm/                 # LSTM metrics
│   │       └── xgboost/              # XGBoost metrics
│   │
│   ├── lstm/                          # LSTM pipeline scripts
│   │   ├── bronze_ncmapss.py
│   │   ├── silver_ncmapss.py
│   │   ├── gold_label_base_ncmapss.py
│   │   ├── gold_label_lstm_ncmapss.py
│   │   ├── gold_feature_lstm_ncmapss.py
│   │   ├── inference_lstm_ncmapss.py
│   │   ├── monitor_lstm_ncmapss.py
│   │   └── utils/                     # LSTM utility modules
│   │
│   └── xgboost/                       # XGBoost pipeline scripts
│       ├── bronze_ncmapss.py
│       ├── silver_ncmapss.py
│       ├── gold_label_base_ncmapss.py
│       ├── gold_label_xgboost_ncmapss.py
│       ├── gold_feature_xgboost_ncmapss.py
│       ├── inference_xgboost_ncmapss.py
│       ├── monitor_xgboost_ncmapss.py
│       └── utils/                     # XGBoost utility modules
│
├── docs/                              # Documentation
│   ├── 4_stage_dag_architecture.md
│   ├── quick_start_guide.md
│   ├── execution_guide.md
│   └── ... (additional guides)
│
├── notebooks/                         # Development notebooks
│   └── N CMAPSS ML Model.ipynb       # Main training notebook
│
├── docker-compose.yaml               # Docker orchestration
├── Dockerfile                        # Custom Airflow image
├── requirements.txt                  # Python dependencies
├── README.md                         # Project overview
└── ARCHITECTURE.md                   # This file
```

---

## Data Flow

### End-to-End Data Flow Diagram

```
┌──────────────┐
│  Raw H5 Data │  (scripts/data/*.h5)
│ ~350MB/month │
└──────┬───────┘
       │
       │ [Stage 1: Ingestion]
       │ • Extract arrays from HDF5
       │ • Preserve unit sequences
       │ • Convert to Parquet
       ▼
┌──────────────┐
│ Bronze Layer │  (datamart/bronze/n_cmapss/)
│ ~1.75GB/month│
└──────┬───────┘
       │
       │ [Stage 2: Preprocessing - Silver]
       │ • Normalize sensor values
       │ • Handle missing data
       │ • Quality checks
       ▼
┌──────────────┐
│ Silver Layer │  (datamart/silver/n_cmapss/)
│ ~1.78GB/month│
└──────┬───────┘
       │
       │ [Stage 2: Preprocessing - Gold Label Base]
       │ • Clip RUL at 90 cycles
       │ • Rename columns
       │ • Calculate max cycles
       ▼
┌───────────────┐
│ Gold Label    │  (datamart/gold/label_base/n_cmapss/)
│ Base (Shared) │
└───┬───────┬───┘
    │       │
    │       └──────────────────────┐
    │                              │
    │ [LSTM Path]                  │ [XGBoost Path]
    │                              │
    ▼                              ▼
┌─────────────┐              ┌──────────────┐
│ Gold LSTM   │              │ Gold XGBoost │
│ Labels      │              │ Labels       │
└──────┬──────┘              └──────┬───────┘
       │                            │
       ▼                            ▼
┌─────────────┐              ┌──────────────┐
│ Gold LSTM   │              │ Gold XGBoost │
│ Features    │              │ Features     │
│ (12 sens)   │              │ (96 feat)    │
└──────┬──────┘              └──────┬───────┘
       │                            │
       │ [Stage 3: Inference]       │
       │ • Load model               │
       │ • Scale features           │
       │ • Generate predictions     │
       ▼                            ▼
┌─────────────┐              ┌──────────────┐
│ Inference   │              │ Inference    │
│ LSTM        │              │ XGBoost      │
└──────┬──────┘              └──────┬───────┘
       │                            │
       │ [Stage 4: Monitoring]      │
       │ • Calculate metrics        │
       │ • Detect drift             │
       │ • Generate reports         │
       ▼                            ▼
┌─────────────┐              ┌──────────────┐
│ Monitoring  │              │ Monitoring   │
│ LSTM        │              │ XGBoost      │
└─────────────┘              └──────────────┘
```

---

## Component Specifications

### Bronze Layer Processing

**Script**: `bronze_ncmapss.py`

**Input**:

- H5 file path: `scripts/data/N-CMAPSS_DS_{date}.h5`
- Snapshot date: YYYY-MM-DD

**Processing**:

1. Open HDF5 file
2. Read dev/test splits
3. Extract arrays: W, X_s, X_v, T, Y, A
4. Concatenate dev + test
5. Create DataFrame
6. Write to Parquet (single file per snapshot)

**Output**:

- Path: `datamart/bronze/n_cmapss/snapshot_date={date}/data.parquet`
- Format: Parquet (snappy compression)
- Size: ~1.75 GB per snapshot

**Performance**:

- Runtime: 2-3 minutes per snapshot
- Memory: ~4GB peak

---

### Silver Layer Processing

**Script**: `silver_ncmapss.py`

**Input**:

- Bronze parquet: `datamart/bronze/n_cmapss/snapshot_date={date}/`

**Processing** (PySpark):

1. Read Bronze Parquet
2. Explode nested arrays to columns
3. Apply z-score normalization to sensors
4. Filter invalid values
5. Add formatted unit IDs (DS01_001 format)
6. Partition by snapshot_date

**Output**:

- Path: `datamart/silver/n_cmapss/snapshot_date={date}/`
- Format: Partitioned Parquet
- Size: ~1.78 GB per snapshot

**Performance**:

- Runtime: 3-5 minutes per snapshot
- Memory: 4GB driver, 4GB executor

---

### Gold Label Base Processing

**Script**: `gold_label_base_ncmapss.py`

**Input**:

- Silver parquet: `datamart/silver/n_cmapss/snapshot_date={date}/`

**Processing**:

1. Read Silver data
2. Calculate max cycle per unit
3. Clip RUL at 90 cycles: `hs = min(max_cycle - cycle, 90)`
4. Rename columns to descriptive names
5. Select 12 key sensor features

**Output**:

- Path: `datamart/gold/label_base/n_cmapss/snapshot_date={date}/`
- Format: Partitioned Parquet
- Size: ~500 MB per snapshot

**Performance**:

- Runtime: 1-2 minutes per snapshot
- Memory: ~2GB

---

### LSTM Feature Engineering

**Script**: `gold_feature_lstm_ncmapss.py`

**Input**:

- Gold label base: `datamart/gold/label_base/n_cmapss/snapshot_date={date}/`

**Processing**:

1. Read label base Parquet
2. Select 12 sensor features
3. Convert to Darts TimeSeries objects (one per unit)
4. Save as pickle files

**Output**:

- Path: `datamart/gold/feature/lstm/n_cmapss/snapshot_date={date}/`
- Files:
  - `covariates.pkl` (list of TimeSeries)
  - `units.pkl` (list of unit IDs)
- Size: ~200 MB per snapshot

**Performance**:

- Runtime: 2-3 minutes per snapshot
- Memory: ~3GB

---

### XGBoost Feature Engineering

**Script**: `gold_feature_xgboost_ncmapss.py`

**Input**:

- Gold label base: `datamart/gold/label_base/n_cmapss/snapshot_date={date}/`

**Processing**:

1. Read label base Parquet
2. For each of 12 base features:
   - Calculate rolling mean (5, 15, 30 cycles)
   - Calculate rolling std (5, 15, 30 cycles)
   - Calculate acceleration (diff of rolling mean)
3. Handle edge cases (NaN fill with base feature)
4. Total: 12 base + 84 engineered = 96 features

**Output**:

- Path: `datamart/gold/feature/xgboost/n_cmapss/snapshot_date={date}/`
- Format: Parquet
- Size: ~800 MB per snapshot

**Performance**:

- Runtime: 5-10 minutes per snapshot
- Memory: ~6GB peak

---

### LSTM Inference

**Script**: `inference_lstm_ncmapss.py`

**Input**:

- LSTM features: `datamart/gold/feature/lstm/n_cmapss/snapshot_date={date}/`
- LSTM labels: `datamart/gold/label/lstm/n_cmapss/snapshot_date={date}/`
- Model: `model_bank/darts_lstm_model.pkl`
- Scalers: `model_bank/darts_*_scaler.pkl`

**Processing**:

1. Load model and scalers
2. Load TimeSeries features and labels
3. Scale features and labels
4. Run historical_forecasts (rolling predictions)
5. Inverse transform predictions
6. Calculate RMSE/MAE

**Output**:

- Path: `datamart/inference/lstm/n_cmapss/snapshot_date={date}/predictions.parquet`
- Columns: unit, dataset, time, RUL_Predicted, RUL_Actual, Prediction_Error
- Size: ~50 MB per snapshot

**Performance**:

- Runtime: 5-15 minutes per snapshot
- Memory: ~4GB (CPU) or 8GB (GPU)
- GPU: 2-3x faster if available

---

### XGBoost Inference

**Script**: `inference_xgboost_ncmapss.py`

**Input**:

- XGBoost features: `datamart/gold/feature/xgboost/n_cmapss/snapshot_date={date}/`
- XGBoost labels: `datamart/gold/label/xgboost/n_cmapss/snapshot_date={date}/`
- Model: `model_bank/xgboost_rul_model.pkl`

**Processing**:

1. Load XGBoost model
2. Load features and labels
3. Run batch predictions
4. Calculate RMSE/MAE

**Output**:

- Path: `datamart/inference/xgboost/n_cmapss/snapshot_date={date}/predictions.parquet`
- Columns: unit, dataset, cycle, RUL_Predicted, RUL_Actual, Prediction_Error
- Size: ~100 MB per snapshot

**Performance**:

- Runtime: 2-5 minutes per snapshot
- Memory: ~3GB

---

### Monitoring

**Scripts**:

- `monitor_lstm_ncmapss.py`
- `monitor_xgboost_ncmapss.py`

**Input**:

- Inference results: `datamart/inference/{lstm|xgboost}/n_cmapss/snapshot_date={date}/`

**Processing**:

1. Load predictions
2. Calculate overall metrics (RMSE, MAE)
3. Calculate per-dataset/per-unit metrics
4. Compare against baseline (first month)
5. Detect drift (statistical tests)
6. Generate visualizations (matplotlib)
7. Save JSON reports

**Output**:

- Path: `datamart/monitoring/{lstm|xgboost}/n_cmapss/snapshot_date={date}/`
- Files:
  - `{model}_monitoring_report.json` (metrics)
  - `performance_chart.png` (visualizations)
- Size: ~1-5 MB per snapshot

**Performance**:

- Runtime: 1-2 minutes per snapshot
- Memory: ~2GB

---

## Deployment Instructions

### Initial Setup

```bash
# 1. Clone repository
cd "F:\MITB Projects\MLE Proj LSTM"

# 2. Ensure raw data exists
ls scripts/data/*.h5

# 3. Start Docker services
docker-compose up -d

# 4. Wait for initialization
docker logs mleprojlstm-airflow-scheduler-1 --follow

# 5. Access Airflow UI
open http://localhost:8080
# Username: admin
# Password: admin

# 6. Verify DAGs are loaded
# All 8 DAGs should appear in the UI

# 7. Enable DAGs
# Click the toggle switch for each DAG

# 8. Monitor execution
# DAGs will run monthly with catchup enabled
```

### Scaling Considerations

**Horizontal Scaling**:

- Add Airflow workers with Celery executor
- Use PostgreSQL for task queue (current: SQLite)
- Deploy on Kubernetes for auto-scaling

**Vertical Scaling**:

- Increase Docker memory limit (16GB recommended)
- Allocate more CPU cores to Spark executor
- Use GPU-enabled container for LSTM

**Data Partitioning**:

- Process datasets independently (DS01-DS08)
- Parallel processing of monthly snapshots
- Batch inference by engine units

---

## Security Considerations

- **Airflow Credentials**: Change default admin/admin password
- **Data Access**: Restrict volume mount permissions
- **Model Files**: Secure model_bank directory
- **Network**: Use internal Docker network for inter-service communication
- **Secrets**: Use Airflow Connections for sensitive configs

---

## Monitoring & Observability

**Airflow Logs**:

```bash
# View scheduler logs
docker logs mleprojlstm-airflow-scheduler-1 --follow

# View task logs
docker exec mleprojlstm-airflow-scheduler-1 \
    airflow tasks logs {dag_id} {task_id} {execution_date}
```

**Performance Metrics**:

- Task duration: Airflow UI → Graph View
- Data sizes: Check datamart directories
- Model metrics: Review monitoring JSON reports

**Alerts**:

- Configure Airflow email alerts on task failure
- Set up threshold alerts for model performance degradation
- Monitor disk space usage

---

## Troubleshooting

Common issues and solutions are documented in `README.md` under the Troubleshooting section.

---

**Document Version**: 1.0
**Last Updated**: Nov 2025
**Maintained By**: Justin Ng from Group 10
