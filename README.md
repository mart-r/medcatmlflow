# The _medcatmlflow_ package is designed to measure metrics of and serve MedCAT models.

It uses `mlflow` for model tracking and serving.

It uses regression tooling within `medcat` to generate metrics.


# Installing _medcatmlflow_

We've packaged the project into a docker container.

## Running in development mode
```
docker-compose -f docker-compose-dev.yml up
```
Or with the `-d` option to run it in detached mode.

## Running in production mode
In production, we want to use a specific pre-built image.
That's why we use the `docker-compose-prod.yml` instead.

The steps are as follows:
1. Get the `docker-compose-prod.yml`
  - Either by cloning `git clone -b nearProduction --single-branch git@github.com:mart-r/medcatmlflow.git/`
  - Or by copying the contents of the file (i.e if github is not available)
2. Setup configs
  - \[Optional\] Change some of the environmental variables in `docker-compose-prod.yml` to suit your needs / environment
    - You can change where the models (`MEDCATMLFLOW_MODEL_STORAGE_PATH`) or the database (`MEDCATMLFLOW_DB_URI`) are saved
    - You can change the log path (`MEDCATMLFLOW_LOGS_PATH`) and level (`MEDCATMLFLOW_LOGS_LEVEL`)
    - You can change the MedCATtrainer URL (`MCT_BASE_URL`)
  - \[Optional\] You can specify MedCATtrainer login details in `.env`
3. Run the container
  - `docker-compose -f docker-compose-prod.yml up -d`


# How to use _medcatmlflow_

When the service is running, you just need to go to [http://localhost:8000/](http://localhost:8000/) (by default).
You can then start uploading models and looking at the model hierarchies.
