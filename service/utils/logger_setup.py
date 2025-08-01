import logging
import os
import datetime


def setup_logging(log_dir="logs"):
    """
    Set up logging to file and console. Creates a new log file for each run.
    Returns the log file path.
    """
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{run_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return log_file
