# Aircraft Engine Predictive Maintenance - N-CMAPSS Dataset

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-20.10+-blue.svg)](https://www.docker.com/)
[![Airflow](https://img.shields.io/badge/Airflow-2.11.0-green.svg)](https://airflow.apache.org/)

> Production-grade MLOps pipeline for predicting Remaining Useful Life (RUL) of aircraft engines using deep learning (LSTM) and gradient boosting (XGBoost) models.

---

## 📋 Overview

### What is This Project?

This system implements **Remaining Useful Life (RUL) prediction** for aircraft engines using NASA's N-CMAPSS dataset. Two machine learning approaches are compared:

1. **LSTM (Long Short-Term Memory)** - Deep learning with 30-cycle sequences
2. **XGBoost** - Gradient boosting with 96 engineered features

### Key Features

- ✅ **Production-Ready**: Apache Airflow orchestration, Docker containerization
- ✅ **Medallion Architecture**: Bronze → Silver → Gold data layers
- ✅ **Automated Pipelines**: 4-stage DAGs (Ingestion, Preprocessing, Training, Monitoring)
- ✅ **Model Monitoring**: Drift detection, performance tracking, automated reports
- ✅ **Scalable**: Spark for data processing, monthly incremental updates

### System Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 10/11, Linux, macOS |
| **Docker** | Desktop 20.10+ |
| **RAM** | 16GB minimum |
| **Storage** | 50GB free space |

---

## 🚀 Quick Start

### 1. Prerequisites

- Docker Desktop installed and running
- Raw data files in `scripts/data/`: `N-CMAPSS_DS_2025-01-01.h5` through `N-CMAPSS_DS_2025-10-01.h5`

### 2. Start Pipeline

```powershell
cd "F:\MITB Projects\MLE Proj LSTM"
docker-compose up -d
docker ps  # Verify 3 containers running
```

### 3. Access Airflow UI

- URL: **http://localhost:8080**
- Username: `admin` / Password: `admin`
- Enable all 8 DAGs

### 4. Monitor Progress

```powershell
ls scripts\datamart\bronze\n_cmapss\     # Bronze layer
ls scripts\datamart\inference\lstm\      # LSTM predictions
ls scripts\datamart\inference\xgboost\   # XGBoost predictions
```

**Runtime**: 3-6 hours (first run), 20-40 minutes (monthly updates)

---

## 📐 Architecture

For complete system architecture, see **[ARCHITECTURE.md](ARCHITECTURE.md)**

### High-Level Flow

```
Raw H5 → Bronze (Parquet) → Silver (Cleaned) → Gold (Features+Labels)
    ├─→ LSTM (12 sensors, TimeSeries) → Inference → Monitoring
    └─→ XGBoost (96 features) → Inference → Monitoring
```

### Pipeline Stages

| Stage | Purpose | Runtime/Month |
|-------|---------|---------------|
| **1. Ingestion** | H5 → Parquet | 2-3 min |
| **2. Preprocessing** | Clean & Feature Engineering | 10-15 min |
| **3. Inference** | Model Predictions | 10-20 min |
| **4. Monitoring** | Performance Tracking | 1-2 min |

---

## 📁 Project Structure

```
MLE Proj LSTM/
├── dags/                  # Airflow DAG definitions (8 files)
├── scripts/
│   ├── data/              # Raw H5 files (~3.5GB)
│   ├── model_bank/        # Trained models & scalers
│   ├── datamart/          # Medallion architecture (~50GB)
│   ├── lstm/              # LSTM pipeline scripts
│   └── xgboost/           # XGBoost pipeline scripts
├── docs/                  # Documentation
├── ARCHITECTURE.md        # System design (detailed)
├── README.md              # This file (overview)
└── docker-compose.yaml    # Orchestration config
```

---

## 📊 Model Performance

| Model | Features | RMSE | MAE | Training Time |
|-------|----------|------|-----|---------------|
| **LSTM** | 12 sensors, 30-cycle sequences | 0.10-0.15 | 0.03-0.05 | 2-4 hours (GPU) |
| **XGBoost** | 96 engineered features | 0.08-0.15 | 0.03-0.05 | 15-30 min (CPU) |

---

## 🛠️ Troubleshooting

### Port 8080 Already in Use

```powershell
netstat -ano | findstr :8080
taskkill /PID <process_id> /F
docker-compose up -d
```

### Memory Error

1. Docker Desktop → Settings → Resources → Memory: 16GB+
2. Restart Docker
3. Clear failed tasks:
   ```bash
   docker exec mleprojlstm-airflow-scheduler-1 airflow tasks clear <dag_id> --yes --only-failed
   ```

### Files Not Appearing

Ensure Docker has file sharing enabled for project directory.

### View Logs

```bash
docker logs mleprojlstm-airflow-scheduler-1 --follow
```

For more issues, see full [README.md](README.md) Troubleshooting section.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **README.md** | Project overview (original, comprehensive) |
| **README_NEW.md** | This file - streamlined quick reference |
| **ARCHITECTURE.md** | Complete system design and specifications |
| **CLEANUP_SUMMARY.md** | Code cleanup work summary |
| **docs/** | Detailed guides and methodologies |

---

## 👥 Team

**MITB Project Team**:
1. CHEN Tiancheng
2. CHEN Zhiyang
3. LIN Xiongqing
4. Justin NG
5. SIM Kim Sia

---

## 📄 License

MIT License - See LICENSE file for details.

---

**Last Updated**: January 2025  
**Version**: 2.0 (Production MLOps Pipeline)

**Happy Predicting! 🚀✈️**
