"""
XGBoost Pipeline - Stage 1: Data Ingestion
=========================================
This DAG handles raw data loading from H5 files into the Bronze layer.
Runs daily to ingest new snapshot data.
Datasets: N-CMAPSS (NASA Commercial Modular Aero-Propulsion System Simulation)
"""
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator

from dag_config import DEFAULT_ARGS, SCHEDULE_INTERVAL, START_DATE, END_DATE, CATCHUP

default_args = {
    **DEFAULT_ARGS,
    "depends_on_past": False,
}

with DAG(
    "xgboost_01_ingestion",
    default_args=default_args,
    description="XGBoost Pipeline Stage 1: Raw data ingestion to Bronze layer",
    schedule_interval=SCHEDULE_INTERVAL,
    start_date=START_DATE,
    end_date=END_DATE,
    catchup=CATCHUP,
    tags=["xgboost", "ingestion", "bronze"],
) as dag:
    start = DummyOperator(task_id="start_ingestion")
    bronze_ncmapss = BashOperator(
        task_id="bronze_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts/xgboost && "
            "python3 bronze_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )
    end = DummyOperator(task_id="end_ingestion")
    start >> bronze_ncmapss >> end