"""
XGBoost Pipeline - Stage 3: Training & Inference
================================================
This DAG handles model training (if needed) and inference.
Training is skipped if model exists in model_bank.
Inference only runs for dates >= 2025-01-01 (after training period).
Schedule: Daily, triggered after preprocessing completes
Model: XGBoost with 96 features, RobustScaler normalization
"""
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import ShortCircuitOperator, BranchPythonOperator
import os
import glob

from dag_config import DEFAULT_ARGS, SCHEDULE_INTERVAL, START_DATE, END_DATE, CATCHUP
def should_run_training(**kwargs):
    """
    Check if training should run.
    Training only runs if model doesn't exist.
    Returns:
        bool: True if model is missing, False if model exists
    """
    model_path = "/opt/airflow/scripts/model_bank/xgboost_rul_model.pkl"
    if os.path.exists(model_path):
        print(f"✓ Model exists at {model_path}")
        print(f"  Skipping training - using existing model")
        return False
    else:
        print(f"❌ Model not found at {model_path}")
        print(f"  Training required")
        return True
def should_run_inference(ds, **kwargs):
    """
    Check if inference should run based on:
    1. Date >= 2025-01-01 (after training period)
    2. Model exists
    3. Feature store has data for this snapshot date
    Returns:
        bool: True if inference should run, False to skip
    """
    from datetime import datetime
    execution_date = datetime.strptime(ds, "%Y-%m-%d")
    cutoff_date = datetime(2024, 12, 31)
    print(f"\n{'='*60}")
    print("Checking inference prerequisites...")
    print(f"{'='*60}")
    print(f"Execution date: {execution_date.strftime('%Y-%m-%d')}")
    print(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
    if execution_date < cutoff_date:
        print(f"❌ Date check FAILED: {execution_date.strftime('%Y-%m-%d')} < {cutoff_date.strftime('%Y-%m-%d')}")
        return False
    else:
        print(f"✓ Date check PASSED: {execution_date.strftime('%Y-%m-%d')} >= {cutoff_date.strftime('%Y-%m-%d')}")
    model_path = "/opt/airflow/scripts/model_bank/xgboost_rul_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model check FAILED: Model not found at {model_path}")
        return False
    else:
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model check PASSED: Found model at {model_path} ({model_size_mb:.2f} MB)")
    feature_dir = f"/opt/airflow/scripts/datamart/gold/feature/xgboost/n_cmapss/snapshot_date={ds}"
    if not os.path.exists(feature_dir):
        print(f"❌ Feature check FAILED: Feature directory not found at {feature_dir}")
        return False
    partition_dirs = glob.glob(os.path.join(feature_dir, "dataset=*"))
    single_file = os.path.join(feature_dir, "features.parquet")
    if partition_dirs or os.path.exists(single_file):
        print(f"✓ Feature check PASSED: Found features at {feature_dir}")
    else:
        print(f"❌ Feature check FAILED: No feature files found")
        return False
    print(f"\n{'='*60}")
    print("✅ All checks PASSED - Proceeding with inference")
    print(f"{'='*60}\n")
    return True
default_args = {
    **DEFAULT_ARGS,
    "depends_on_past": True,  # Wait for preprocessing to complete
}

with DAG(
    "xgboost_03_training_inference",
    default_args=default_args,
    description="XGBoost Pipeline Stage 3: Model training (if needed) and inference",
    schedule_interval=SCHEDULE_INTERVAL,
    start_date=START_DATE,
    end_date=END_DATE,
    catchup=CATCHUP,
    tags=["xgboost", "training", "inference"],
) as dag:
    start = DummyOperator(task_id="start_training_inference")
    def branch_training_or_skip(**context):
        """Decides whether to run training or skip it"""
        if should_run_training(**context):
            return "train_xgboost_model"  # Run training
        else:
            return "skip_training"  # Skip training
    check_training_needed = BranchPythonOperator(
        task_id="check_training_needed",
        python_callable=branch_training_or_skip,
        provide_context=True,
    )
    # Training: Only runs if model doesn't exist
    # NOTE: This is a placeholder - actual training should be done in notebook
    # and models/scalers copied to model_bank
    train_model = BashOperator(
        task_id="train_xgboost_model",
        bash_command=(
            "echo '⚠️  Training placeholder - run notebook training and copy:' && "
            "echo '   1. xgboost_rul_model.pkl' && "
            "echo '   2. xgboost_feature_scaler.pkl' && "
            "echo '   to /opt/airflow/scripts/model_bank/' && "
            "exit 1"  # Fail to indicate manual action required
        ),
    )
    # Dummy task for when training is skipped
    skip_training = DummyOperator(task_id="skip_training")
    # Join point after training branch - ensures inference runs regardless of training path
    training_complete = DummyOperator(
        task_id="training_complete",
        trigger_rule="none_failed_min_one_success"  # Continue if either training or skip succeeds
    )
    # Check if inference should run
    check_inference_prerequisites = ShortCircuitOperator(
        task_id="check_inference_prerequisites",
        python_callable=should_run_inference,
        provide_context=True,
    )
    # Inference: XGBoost RUL predictions with RobustScaler
    inference_xgboost = BashOperator(
        task_id="inference_xgboost_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts/xgboost && "
            "python3 inference_xgboost_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )
    end = DummyOperator(task_id="end_training_inference")
    # Pipeline flow
    # Training branch (only if model missing)
    start >> check_training_needed
    check_training_needed >> train_model >> training_complete
    check_training_needed >> skip_training >> training_complete
    # Inference branch (after training check)
    training_complete >> check_inference_prerequisites >> inference_xgboost >> end