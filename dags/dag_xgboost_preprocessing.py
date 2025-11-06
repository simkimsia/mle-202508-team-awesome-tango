"""
XGBoost Pipeline - Stage 2: Data Preprocessing
==============================================
This DAG handles data cleaning, feature engineering, and gold table creation.
Processes Bronze → Silver → Gold (Label + Features).
Schedule: Daily, triggered after ingestion completes
Features: 96 engineered features (12 base + rolling windows)
Scaling: RobustScaler applied during training, not preprocessing
"""
from datetime import timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator

from dag_config import DEFAULT_ARGS, SCHEDULE_INTERVAL, START_DATE, END_DATE, CATCHUP

default_args = {
    **DEFAULT_ARGS,
    "depends_on_past": True,  # Wait for ingestion to complete
}

with DAG(
    "xgboost_02_preprocessing",
    default_args=default_args,
    description="XGBoost Pipeline Stage 2: Data cleaning and feature engineering",
    schedule_interval=SCHEDULE_INTERVAL,
    start_date=START_DATE,
    end_date=END_DATE,
    catchup=CATCHUP,
    tags=["xgboost", "preprocessing", "silver", "gold"],
) as dag:
    start = DummyOperator(task_id="start_preprocessing")
    silver_ncmapss = BashOperator(
        task_id="silver_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts/xgboost && "
            "python3 silver_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )
    gold_label_base = BashOperator(
        task_id="gold_label_base_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts/xgboost && "
            "python3 gold_label_base_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )
    gold_label_xgboost = BashOperator(
        task_id="gold_label_xgboost_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts/xgboost && "
            "python3 gold_label_xgboost_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )
    gold_feature_xgboost = BashOperator(
        task_id="gold_feature_xgboost_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts/xgboost && "
            "python3 gold_feature_xgboost_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
        execution_timeout=timedelta(hours=2),
    )
    end = DummyOperator(task_id="end_preprocessing")
    start >> silver_ncmapss >> gold_label_base >> gold_label_xgboost >> gold_feature_xgboost >> end