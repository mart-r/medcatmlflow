FROM python:3.10

RUN echo "Wazzup"

COPY src /app/src

COPY mlruns /app/mlruns

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["mlflow", "ui", "--host", "0.0.0.0"]