
# MLOps Release plan

This is a proposed release plan to track exisiting and upcoming features of the MLOps pipeline.




## version 1.0 - Release date: August 2021 :heavy_check_mark:
- CI in Azure DevOps
- Unit testing results displayed on CI build
    - Basic tests
- Linting results displayed on CI build
- Train in AML pipeline
- Model evaluation in AML pipeline
- Registration in AML pipeline
- AML model registration triggers CD through in Azure DevOps

## version 2.0 - Release date: TBD
- Data Drift trigger
- Multiple model deployment
- Unit testing results displayed on CI build
    - Environment testing with tox
    - Type testing with mypy
- Linting score threshold on CI build
- Dev / Prod entry scripts and deployments

## version 3.0 - Release date: TBD
- ETL in AML pipeline
- Unit testing library
- Parallel Hyperparameter Optimisation
