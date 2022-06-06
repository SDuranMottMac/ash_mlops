"""
    This script step is where we initialise and train our model. This example is likely more complex
    than most as it is an ensemble model, albeit a rather simple one.

    However, we expect that this script will take in a training dataset, and return a pickled model.

    The training dataset is a tabular AML dataset.

    The basic steps you might want to follow are:
    1. Initialise model and set hyperparameters
    2. Train the model on the training data (using `model.fit()` or something similar)
    3. Pickle the model and save it to the run outputs.
    
    # should-change refer to the README doc, you should have a script named
    train.py and you can use either this file or train_file as your
    skeleton for that file.

"""
# pipeline imports
import os
from typing import Dict, List, Union
import dill
from azureml.core.run import Run

# model imports
from sklearn.metrics import mean_squared_error as mse
from statsmodels.tsa.statespace.sarimax import SARIMAX as arimax
import xgboost as xgb


class Model:
    """
    An Ensemble Model for Predicting Canal Level.

    Uses statsmodels SARIMAX and xgboost XGBRegressor, the former is used to predict overall trends,
    and the latter is used for the perturbations caused by severe rain events.

    Expected Dataset:
    time, gauge_level, rain_window_1, ..., rain_window_n
    2018-01-11 05:00:00, 0.6804123711340208, 0.0, 0.1376958655330867
    ...
    """

    def __init__(
        self, hyperparams: Dict[str, Union[int, float, List[Union[int, float]]]]
    ):
        """
        Initializes ensemble model

        Args:
            hyperparams (Dict[str, Union[int, float, List[Union[int, float]]]]): hyperparameters
            used for both models, of the form:
            {
                'xgb_model_n_estimators': int,
                'xgb_model_reg_lambda': int,
                'xgb_model_gamma': int,
                'xgb_model_max_depth': int,
                'arima_model_order': list,
                'arima_model_seasonal_order': list,
                'arima_model_max_iter': int,
            }
        """
        self.hyperparams = hyperparams

        self.arima = None
        self.xgb_reg = None
        self.stats = None

    def train(self, data, stats=True) -> Union[str, Dict]:
        """
        Trains both the arima and xgboost models.

        Args:
            data ([type]): data needed to train the models
            stats (bool, optional): if true will save and return the stats
            attribute in the object, if false won't do that. Defaults to True.

        Returns:
            [Union[str, Dict]]: either returns 'done!' if stats is false or
            self.stats
        """
        self.arima, arima_mse = self.train_arimax_model(data)
        self.xgb_reg, xgb_mse = self.train_xgb_model(data)

        if stats:
            self.stats = {
                "arima_in_sample_mse": arima_mse,
                "xgb_in_sample_mse": xgb_mse,
                "combined_in_sample_mse": mse(self.predict(data), data["gauge_level"]),
            }
            return self.stats

        else:
            return "done!"

    def calc_coeffs(self) -> Dict[str, float]:
        """
        Returns the weights of the xgboost and arima models

        Returns:
            Dict[str, float]: dict of the form
                {'xgb': float, 'arima': float}
        """
        # will make these variable in the future
        return {"xgb": 0.12800904, "arima": 0.87219783}

    def predict(self, data):
        """
        Returns prediction using the ensemble model

        Args:
            data ([type]): rain data used to predict
            result.

        Returns:
            [type]: prediction
        """
        # get rain data features
        features = self.get_features(data)

        # note that as we are using a AR timeseries model, we need gauge_level
        # for at least the start of our prediction period
        arima_prediction = self.arima.apply(
            endog=data["gauge_level"], exog=data[features]
        ).fittedvalues

        xgb_prediction = self.xgb_reg.predict(data[features])

        coeffs = self.calc_coeffs()

        return coeffs["arima"] * arima_prediction + coeffs["xgb"] * xgb_prediction

    def get_features(self, data) -> List[str]:
        """
        Gets relevant rain data columns from
        data.

        Args:
            data ([type]): data given

        Returns:
            List[str]: list of relevant columns
        """
        return [col for col in data.columns if col[:12] == "rain_window_"]

    def score(self, data):
        """
        Scores prediction using mean squared error

        Args:
            data ([type]): [description]

        Returns:
            [type]: mean squared error between prediction and actual
        """
        prediction = self.predict(data)
        return mse(prediction, data["gauge_level"])

    def train_arimax_model(self, data):
        """
        Trains the Arimax model

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        # get rain data
        features = self.get_features(data)

        # define model
        model = arimax(
            endog=data.gauge_level,
            exog=data[features],
            order=self.hyperparams["arima_model_order"],
            seasonal_order=self.hyperparams["arima_model_seasonal_order"],
        )

        # train!
        model = model.fit(maxiter=self.hyperparams["arima_model_max_iter"])

        return model, model.mse

    def train_xgb_model(self, data):
        """
        Trains the xgboost model

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        # shuffle
        data = data.sample(frac=1)

        # define model
        xgb_reg = xgb.XGBRegressor(
            n_estimators=self.hyperparams["xgb_model_n_estimators"],
            reg_lambda=self.hyperparams["xgb_model_reg_lambda"],
            gamma=self.hyperparams["xgb_model_gamma"],
            max_depth=self.hyperparams["xgb_model_max_depth"],
        )

        # get rain data
        features = self.get_features(data)

        # train!
        xgb_reg.fit(data[features], data["gauge_level"])

        # in sample evaluation
        xgb_reg_test = xgb_reg.predict(data[features])

        return xgb_reg, mse(xgb_reg_test, data["gauge_level"])

    def hyperparams_check(self) -> bool:
        """
        Makes sure that the hyperparameters inserted are of the correct format

        Returns:
            Boolean: Returns true if everything is fine, false if there is an
            error.
            can return None and use Exceptions
            but i prefer to have it this way for testing purposes

        TODO: nested typings checks
        passing on true exception without using exceptions
        """
        if not hasattr(self, "hyperparams"):
            return False
            # raise Exception("No hyperparams to use - hyperparams are not an attribute")

        expected_hyperparams = {
            "xgb_model_n_estimators": int,
            "xgb_model_reg_lambda": int,
            "xgb_model_gamma": int,
            "xgb_model_max_depth": int,
            "arima_model_order": list,
            "arima_model_seasonal_order": list,
            "arima_model_max_iter": int,
        }

        for exp_h, typ_h in expected_hyperparams.items():
            if exp_h not in self.hyperparams:
                return False
                # raise Exception("No Hyperparameter for " + str(exp_h))
            if not isinstance(self.hyperparams[exp_h], typ_h):
                return False
                # raise Exception("Type of hyperparameter " + str(exp_h) + " is invalid")

        return True


run = Run.get_context()

if __name__ == "__main__":

    # initialise model
    model_instance = Model(
        {
            "xgb_model_n_estimators": 100,
            "xgb_model_reg_lambda": 1,
            "xgb_model_gamma": 0,
            "xgb_model_max_depth": 64,
            "arima_model_order": [2, 1, 2],
            "arima_model_seasonal_order": [0, 0, 1, 8],
            "arima_model_max_iter": 250,
        }
    )

    # load training and testing datasets
    train = run.input_datasets["train_data"].to_pandas_dataframe()

    # train the model on the data passed as a script argument
    model_instance.train(train)

    # save model.pkl
    model_file_path = run.output_datasets["model"]
    with open(os.path.join(model_file_path, "model.pkl"), "wb") as file:
        dill.dump(model_instance, file)
