#!/bin/bash

# Check if the directory specified by MODEL_STORAGE_PATH exists
if [ ! -d "$MODEL_STORAGE_PATH" ]; then
  # If not, create the directory
  mkdir -p "$MODEL_STORAGE_PATH"
fi

mkdir -p $LOGS_PATH

# Run your application
cd src
python -m flask --app app.app run --host "0.0.0.0"
# python src/app/app.py