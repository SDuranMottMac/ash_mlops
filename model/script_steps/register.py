"""
    This script step is where we register the trained model in our Azure Machine Learning Workspace,
    pending checks performed in evaluate.py.

    We upload a folder containing everything to do with the model to the run history, then we
    register this with Azure ML.

    In this file for a different model we currently need to change the model name and tags.

    Otherwise, unless you have made drastic changes to the evaluation output 'summary.json', you
    should be able to leave this script relatively untouched.

"""

# script imports
#  - os for defining the path of the data and model on AML compute
#  - json for interacting with the evaluate.py output; summary.json
#  - azureml.core.run for moving the model in the pipeline, including registering it to our AML
#    workspace
import os
import json
from azureml.core.run import Run

# get data from the aml pipeline
run = Run.get_context()

if __name__ == "__main__":

    # get the file path of the model input
    model_file_path = run.input_datasets["model"]
    model_score_file_path = run.input_datasets["model_score"]

    # get results from evaluation step
    with open(os.path.join(model_score_file_path, "summary.json"), "r") as file:
        model_score_summary = json.loads(file.read())

    # upload folder containing model to run outputs
    run.upload_folder(name="model", path=os.path.join(model_file_path))

    # upload folder containing model score to run outputs
    run.upload_folder(name="model_score", path=os.path.join(model_score_file_path))

    # register model from run outputs, if it passed evaluation
    # note these tags and name will need to be changed for each model
    if model_score_summary["register_trained_model"]:
        model = run.register_model(
            model_name="model",
            model_path="model",
            tags={
                "area": "your specified area",
                "predicts": "what your model's prediction",
                "using": "your data source type",
            },
        )
