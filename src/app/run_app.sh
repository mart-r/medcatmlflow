#!/bin/bash

# Check if the directory specified by MODEL_STORAGE_PATH exists
if [ ! -d "$MODEL_STORAGE_PATH" ]; then
  # If not, create the directory
  mkdir -p "$MODEL_STORAGE_PATH"
fi

# Run your application
python src/app/app.py