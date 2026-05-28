import os 
import logging

DEBUG = True
CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
LOG_FILE = os.path.join(PROJECT_ROOT, "debugging", "pipeline_logs.log")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", 
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  
    ]
)