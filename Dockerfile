FROM python:3.10

COPY src /app/src

COPY mlruns /app/mlruns

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "src/main.py"]