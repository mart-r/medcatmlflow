FROM python:3.10

# COPY src /app/src

# COPY mlruns /app/mlruns

COPY src/app /app/src/app

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

# RUN mkdir -p $MODEL_STORAGE_PATH

# CMD ["python", "src/app/app.py"]

# Set execute permissions on the script
RUN chmod +x /app/src/app/run_app.sh

# Use the run_app.sh script to start the application
CMD ["src/app/run_app.sh"]
