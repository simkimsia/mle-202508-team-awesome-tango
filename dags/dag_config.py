"""
Shared DAG Configuration
========================
Centralized configuration for all Airflow DAGs.
This ensures consistent scheduling and date ranges across all pipelines.
"""

from datetime import datetime, timedelta

# Schedule Configuration
SCHEDULE_INTERVAL = "0 0 * * *"  # daily at 00:00
START_DATE = datetime(2025, 1, 8)  # default is 1st Jan
END_DATE = datetime(2025, 1, 8)  # cut off at 10th Jan
CATCHUP = True

# Default Arguments
DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
