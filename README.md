# The _medcatmlflow_ package is designed to measure metrics of and serve MedCAT models.

It uses `mlflow` for model tracking and serving.

It uses regression tooling within `medcat` to generate metrics.


# Installing _medcatmlflow_

We've packaged the project into a docker container.

All the user needs to do is run
```
docker-compose up
```
Or with the `-d` option to run it in detached mode.


# How to use _medcatmlflow_


## Prerequisites

In order to run some regression tests you will need to have
- A MedCAT model (version 1.3+ should work)
- At least one regression test suite (multiple recommended)
  - Take a look at the regression test tutorial ([here](https://htmlpreview.github.io/?https://github.com/CogStack/MedCATtutorials/blob/main/notebooks/specialised/Comparing_Models_with_RegressionSuite.html)) for specifics on how to generate these
