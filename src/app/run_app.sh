#!/bin/bash
set -e
set -x

# Check if the directory specified by MODEL_STORAGE_PATH exists
if [ ! -d "$MEDCATMLFLOW_MODEL_STORAGE_PATH" ]; then
  # If not, create the directory
  mkdir -p "$MEDCATMLFLOW_MODEL_STORAGE_PATH"
fi

mkdir -p $MEDCATMLFLOW_LOGS_PATH

# Run your application
cd src
python -m flask --app app.app run --host "0.0.0.0" --without-threads
# python src/app/app.py