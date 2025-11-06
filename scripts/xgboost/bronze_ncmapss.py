from datetime import datetime, timedelta
import argparse
import glob
import os
import pprint
import random

from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils.data_processing_bronze_ncmapss

def main(snapshotdate):
    print('\n\n---starting job---\n\n')
    snapshot_date_str = snapshotdate
    bronze_lms_directory = "../datamart/bronze/"
    if not os.path.exists(bronze_lms_directory):
        os.makedirs(bronze_lms_directory)
    utils.data_processing_bronze_ncmapss.process_bronze_table(
        snapshot_date_str,
        bronze_lms_directory
    )
    print('\n\n---completed job---\n\n')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)