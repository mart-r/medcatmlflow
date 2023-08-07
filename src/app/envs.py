import os

STORAGE_PATH = os.environ.get("MEDCATMLFLOW_MODEL_STORAGE_PATH")

DB_URI = os.environ.get("MEDCATMLFLOW_DB_URI")

LOG_PATH = os.environ.get("MEDCATMLFLOW_LOGS_PATH",
                          os.path.join("..", "..", "logs"))
LOG_BACKUP_DAYS = int(os.environ.get("MEDCATMLFLOW_LOG_BACKUP_DAYS", "30"))
LOG_LEVEL = os.environ.get("MEDCATMLFLOW_LOG_LEVEL", "INFO")

# linking to MedCATtrainer
MCT_USERNAME = os.environ.get("MCT_USERNAME")
MCT_PASSWORD = os.environ.get("MCT_PASSWORD")

MCT_BASE_URL = os.environ.get("MCT_BASE_URL")
