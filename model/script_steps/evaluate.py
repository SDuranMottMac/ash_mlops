"""
    This script step is where we evaluate our trained model, and decide if it is good enough to
    deploy.

    In this example, we are training a canal level model, and we really care about our predictions
    being able to forecast spikes due to heavy precipitation. As such, our conditions for deployment
    are:
      1. for whole dataset: trained model mse < deployed model mse
      2. for dataset with peak rain: trained model mse < deployed model mse

"""
# script imports
#  - os for defining the path of the data and model on AML compute
#  - json for storing the model score, and http requests to deployed model
#  - dill for loading the pickled model
#  - pandas for handling the dataset in and the predictions out of the models
#  - requests for getting data from the previously deployed model
#  - argparse for getting the endpoint of the previously deployed model as an
#    argument
#  - azureml.core.run for getting access to the data moving through the pipeline, the trained model
#    and the evaluation data, as well as uploading the evaluation results
import os
import json
import argparse
from typing import Union
import dill
import pandas
import requests
import pandas as pd
from azureml.core.run import Run

# evaluation metrics:
#  we probably want to import some functions here to score our model - this module (sklearn.metrics)
#  is great: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.metrics import mean_squared_error as mse

# model requirements:
#  we have to explicitly ignore the linting errors here as the linter doesn't recognise that the
#  modules are in use.
# pylint: disable=unused-import
import xgboost as xgb  # noqa: F401
from statsmodels.tsa.statespace.sarimax import SARIMAX as arimax  # noqa: F401

# this endpoint is for the previously deployed model
# set to `n/a` for first run of this pipeline (before you have a previously deployed model)


# could-change if this script is not evaluating your model properly please feel free to
# change it and use your own evaluation functions
class Evaluate:
    def __init__(self):
        """
        Initializes Evaluator
        """
        # `run` is a container for the data travelling through the pipeline including:
        #   - the trained model
        #   - the evaluation dataset
        #   - the outputs of this script, the evaluation results
        self.run = Run.get_context()

    def get_model_endpoint(self) -> str:
        """
        Gets model endpoint from arguments.

        Returns:
            str: Model Endpoint.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-endpoint", type=str, dest="model_endpoint")
        args = parser.parse_args()
        return args.model_endpoint

    def get_existing_model_predictions(self, data: pandas.DataFrame):
        """
        This function is used as a drop in replacement for model.predict (or equivalent depending on
        the model in use). It should take a Pandas DataFrame of the data and return the predictions
        based on the features provided.

        Hopefully this requires minimal updating for a different model, although some work may be
        required to change data types.


        Args:
            data (pandas.DataFrame): [description]

        Raises:
            Exception: [description]
            Exception: [description]
            Exception: [description]

        Returns:
            [type]: [description]
        """
        # attempt to get model results from endpoint
        try:
            res = requests.post(
                self.get_model_endpoint(), json=json.loads(data.to_json())
            )
            res.raise_for_status()

        # if not 2XX code, raise error
        except requests.exceptions.HTTPError as err:
            raise Exception from err

        # parse predictions from response body
        predictions = res.json()

        # assert that predictions are appropriate format
        if not isinstance(predictions, list):
            raise Exception("Predictions not returned as array.")
        if len(predictions) != len(data):
            raise Exception("Number of rows mismatch: len(data) != len(predictions).")

        return predictions

    def summarise_and_score(self, data: pandas.DataFrame):
        """
        This function takes the dataset, including predictions from both of the models, and returns
        a Python dictionary to be stored as a json.

        The JSON/dict must include a key 'register_trained_model' that is true only if we want to
        replace the model in development with the one we have trained.

        This function should be rewritten for each model. Although, the whole sample mse or looking
        at the precision/recall over the whole sample may be adequate for some use cases.

        In this demonstation repository, we are looking at the whole sample mse, as well as the mse
        when the features suggest it's been very rainy recently, as this is when we really care
        about getting our prediction of the canal level right!

        Args:
            data (pandas.DataFrame): [description]

        Returns:
            [type]: [description]
        """
        summary = {}

        # evaluate standard mse
        summary["training_mse_whole_sample"] = mse(
            data.training_prediction, data.gauge_level
        )
        summary["existing_mse_whole_sample"] = mse(
            data.existing_prediction, data.gauge_level
        )

        # measure rain events
        data["rain_summation"] = self.sum_rain_cols(data)

        # filter on sum to get 3% largest
        rain_event_data = self.filter_rs_top_data(data, percentile=0.97)

        # evaluate rain event mse
        summary["training_mse_rain_events"] = mse(
            rain_event_data.training_prediction, rain_event_data.gauge_level
        )
        summary["existing_mse_rain_events"] = mse(
            rain_event_data.existing_prediction, rain_event_data.gauge_level
        )

        # release model if it performs better than previous in both cases
        # explicitly convert to python bool as json.dumps doesn't support numpy booleans
        summary["register_trained_model"] = bool(
            summary["training_mse_rain_events"] < summary["existing_mse_rain_events"]
            and summary["training_mse_whole_sample"]
            < summary["existing_mse_whole_sample"]
        )
        return summary

    def sum_rain_cols(
        self, data: pandas.DataFrame
    ) -> Union[pandas.DataFrame, pandas.Series]:
        """
        This function sums the values of all the rain_window_ columns
        in a dataframe

        Args:
            data (pandas.DataFrame): [description]

        Returns:
            pandas.DataFrame: [description]
        """
        return data.filter(regex=r"(rain_window_)\w+", axis="columns").sum(
            axis="columns"
        )

    def filter_rs_top_data(
        self, data: pandas.DataFrame, percentile: float = 0.97
    ) -> pandas.DataFrame:
        """
        Returns rain_summationdata in the quantile with percentile higher or equal to the
        one provided.

        Args:
            data (pandas.DataFrame): [description]
            percentile (float, optional): [description]. Defaults to 0.97.

        Returns:
            pandas.DataFrame: [description]

        TODO: could take column name as parameter.
        """
        return data[(data.rain_summation >= data.rain_summation.quantile(percentile))]

    def load_tabular_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Loads tabular dataset

        Args:
            dataset_name (str): [description]

        Returns:
            pd.DataFrame: [description]
        """
        return self.run.input_datasets[dataset_name].to_pandas_dataframe()

    def load_file_dataset(self, dataset_name: str) -> str:
        """
        Loads file dataset and returns download path

        Args:
            dataset_name (str): [description]

        Returns:
            pd.DataFrame: [description]
        """
        run = Run.get_context()
        file_dataset = run.input_datasets["train_data"]
        input_path = os.path.dirname(__file__)
        return os.path.dirname(
            file_dataset.download(target_path=input_path, overwrite=False)[0]
        )

    def load_model_in_training(self, model_name: str) -> any:
        """
        Loads model in training given a model name.
        Uses dill to load the model, could be refactored to
        have different loading options.

        Args:
            model_name (str): [description]

        Returns:
            any: [description]
        """
        model_file_path = self.run.input_datasets[model_name]
        # should-change
        with open(os.path.join(model_file_path, "model.pkl"), "rb") as file:
            return dill.load(file)

    def save_model_score(self, filename: str) -> None:
        """
        Saves model score summary.

        Args:
            filename (str): [description]
        """
        model_score_file_path = self.run.output_datasets["model_score"]
        with open(os.path.join(model_score_file_path, filename), "wb") as file:
            file.write(json.dumps(model_score_summary).encode("utf-8"))


if __name__ == "__main__":

    evaluator = Evaluate()

    # load dataset

    # should-change uncomment this if you're using a file dataset
    # download_path = evaluator.load_file_dataset("test_data")
    # raw_data = pd.read_csv(
    #     f"{download_path}/raw_data.csv", index_col="time", parse_dates=True
    # )

    # should-change comment this line if you're using a file dataset, if not
    # keep it as is
    dataset = evaluator.load_tabular_dataset("test_data")

    # load model in training
    model_in_training = evaluator.load_model_in_training("model")

    # get the model in training predictions using dataset
    model_in_training_predictions = model_in_training.predict(dataset)

    # either a url or a 'n/a'
    if evaluator.get_model_endpoint() == "n/a":
        model_score_summary = {"register_trained_model": True}

    else:
        # get the existing model predictions
        existing_model_predictions = evaluator.get_existing_model_predictions(dataset)

        # add predictions to the dataset
        dataset["training_prediction"] = model_in_training_predictions
        dataset["existing_prediction"] = existing_model_predictions

        # get model release stats
        model_score_summary = evaluator.summarise_and_score(dataset)

    # save model score to be used in register.py
    evaluator.save_model_score("summary.json")
