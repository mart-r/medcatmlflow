# STAGE 1

FROM python:3.10 as builder

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

# STAGE 2
FROM python:3.10

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

COPY src/app /app/src/app

WORKDIR /app

EXPOSE 5000

RUN mkdir -p /app/logs

# Set execute permissions on the script
RUN chmod +x /app/src/app/run_app.sh

# Use the run_app.sh script to start the application
CMD ["src/app/run_app.sh"]
