#!/bin/bash
set -e
set -x

# Check if the directory specified by MODEL_STORAGE_PATH exists
if [ ! -d "$MEDCATMLFLOW_MODEL_STORAGE_PATH" ]; then
  # If not, create the directory
  mkdir -p "$MEDCATMLFLOW_MODEL_STORAGE_PATH"
fi

# Check if the directory for test datasets exists
if [ ! -d "$MEDCATMLFLOW_MODEL_STORAGE_PATH/test_datasets" ]; then
  # If not, create the directory
  mkdir -p "$MEDCATMLFLOW_MODEL_STORAGE_PATH/test_datasets"
fi

mkdir -p $MEDCATMLFLOW_LOGS_PATH

# Run your application
cd src
python -m gunicorn -w $MEDCATMLFLOW_GUNICORN_WORKERS -b 0.0.0.0:5000 --timeout $MEDCATMLFLOW_GUNICORN_TIMEOUT app:app
# python -m flask --app app.app run --host "0.0.0.0" --without-threads
