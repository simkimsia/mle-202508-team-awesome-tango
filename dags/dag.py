from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import ShortCircuitOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "dag",
    default_args=default_args,
    description="data pipeline run daily",
    schedule_interval="0 0 * * *",  # At 00:00 on everyday
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 1, 1),
    catchup=True,
) as dag:
    # --- 0. Start / End Markers ---
    start_pipeline = DummyOperator(task_id='start_pipeline')
    end_pipeline = DummyOperator(task_id='end_pipeline')

    # --- 1. Bronze Layer: Data Processing ---
    bronze_data_processing = BashOperator(
        task_id='run_bronze_data_processing',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_table_1.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_data_processing = BashOperator(
        task_id="run_silver_data_processing",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_table_1.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )


    # --- Task Dependencies ---
    start_pipeline >> bronze_data_processing >> silver_data_processing >> end_pipeline
