"""
Contributors: Antoine Chammas

Summary: This file contains basic functions for testing training scripts.
"""

from typing import Dict, List, Union
from model.script_steps import train
from testing import testing_helpers as th
from testing.model.script_steps.test_samples import testing_samples_train as ts

import pytest
import pandas as pd

# could-change this file to implement the tests you need.


def test_model_initialization():
    """
    Fill all_atrs with the string attributes that you want
    to check exist in the model on initialisation.
    Be aware that this is case sensitive.
    """
    all_atrs = ["hyperparams", "arima", "xgb_reg", "stats"]
    model = train.Model(ts.bidh_ensemble_model_hyperparams_sample_0)

    errors = th.check_atr_in_obj(all_atrs, model)

    assert not errors, "Errors Occured:\n{}".format("\n".join(errors))


def test_att_equality():
    """
    Makes sure that attributes are passed correctly to the model.
    Fill in the all_atrs dict with your attributes and values that
    you used to create the model / expect to see on initialisation.
    TODO: Could be merged with test_model_initialization
    """
    model = train.Model(ts.bidh_ensemble_model_hyperparams_sample_0)
    all_atrs = {
        "hyperparams": ts.bidh_ensemble_model_hyperparams_sample_0,
        "arima": None,
        "xgb_reg": None,
        "stats": None,
    }

    errors = th.check_atr_vals_in_obj(all_atrs, model)

    assert not errors, "Errors Occured:\n{}".format("\n".join(errors))


@pytest.mark.skipif(
    not th.check_func_in_obj("hyperparams_check", train.Model({})),
    reason="No function called hyperparams_check",
)
@pytest.mark.parametrize(
    "input_data,expected,reason",
    [
        (ts.bidh_ensemble_model_hyperparams_sample_0, True, "Failed Valid Case 0"),
        (
            ts.bidh_ensemble_model_invalid_hyperparams_sample_0,
            False,
            "Failed Invalid Case 0",
        ),
        (
            ts.bidh_ensemble_model_invalid_hyperparams_sample_1,
            False,
            "Failed Invalid Case 1",
        ),
    ],
)
def test_hyperparameter_checking(input_data: train.Model, expected: bool, reason: str):
    """
    Checks that the hyperparameter checking func in your model is
    working.
    If you have no hyperparameter checking method, this test will
    be ignored.
    Your hyperparameter checking method should be named:
        hyperparams_check
    and should return:
        true if working fine
        false if not
    """
    model = train.Model(input_data)
    assert model.hyperparams_check() == expected, reason


@pytest.fixture()
def train_test_mock_model(mocker) -> train.Model:
    """
    Generates trained mocked model as a pytest fixture.
    This is how 'train_test_mock_model' is used in multiple tests
    after this point.
    The train function and relevant functions inside
    of the model returned by this function
    are mocked.

    Args:
        mocker ([type]): [description]

    Returns:
        train.Model: mocked model
    """
    mocker.patch.object(
        train.Model,
        "train_arimax_model",
        return_value=({}, ts.bidh_ensemble_model_arima_mse_0),
    )
    mocker.patch.object(
        train.Model,
        "train_xgb_model",
        return_value=({}, ts.bidh_ensemble_model_xgb_mse_0),
    )
    mocker.patch.object(
        train.Model, "predict", return_value=ts.bidh_ensemble_model_prediction_0
    )
    model = train.Model(ts.bidh_ensemble_model_hyperparams_sample_0)
    return model


@pytest.mark.skipif(
    not th.check_func_in_obj("train", train.Model({})),
    reason="No function called train",
)
@pytest.mark.parametrize(
    "stats,expected,reason",
    [
        (True, ts.bidh_ensemble_model_stats_0, "Training error while stats = True"),
        (False, "done!", "Training error while stats = False"),
    ],
)
def test_train(
    train_test_mock_model: train.Model,
    stats: bool,
    expected: Union[Dict[str, List[Union[int, float]]], str],
    reason: str,
):
    """
    Checks that the training function of the ensemble model is working.
    This test is specific to the ensemble model made for bangkok-idh
    Please use this as a template to make your tests and not as a
    generic test.
    """
    assert (
        train_test_mock_model.train({"gauge_level": [1, 1, 1, 1]}, stats=stats)
        == expected
    ), reason
    # if stats == True
    # train returns self.stats already but in case
    # a code change happened, we still check that
    # model.stats = expected
    if stats:
        assert (
            train_test_mock_model.stats == expected
        ), "self.stats is not equal to expected"


@pytest.mark.skipif(
    not th.check_func_in_obj("calc_coeffs", train.Model({})),
    reason="No function called calc_coeffs",
)
@pytest.mark.parametrize(
    "expected,reason",
    [
        (
            ts.bidh_ensemble_model_calccoeffs_0,
            "calc_coeffs error, coefficients are different than expected",
        )
    ],
)
def test_calc_coeffs(
    train_test_mock_model: train.Model,
    expected: Dict[str, List[Union[int, float]]],
    reason: str,
):
    """
    Checks that the calc_coeffs function of the ensemble model is working.
    This test is specific to the ensemble model made for bangkok-idh
    Please use this as a template to make your tests and not as a
    generic test.
    """
    assert train_test_mock_model.calc_coeffs() == expected, reason


@pytest.mark.skipif(
    not th.check_func_in_obj("get_features", train.Model({})),
    reason="No function called get_features",
)
@pytest.mark.parametrize(
    "data,expected,reason",
    [
        (
            ts.bidh_ensemble_model_train_data_cols_0,
            ts.bidh_ensemble_model_features_0,
            "get_features is not working for valid data",
        ),
        (
            ts.bidh_ensemble_model_train_data_cols_1,
            ts.bidh_ensemble_model_features_0,
            "get_features is not working for a mix of valid and invalid data",
        ),
        (pd.DataFrame(), [], "get_features is not working for invalid data"),
    ],
)
def test_get_features(
    train_test_mock_model: train.Model,
    data: pd.DataFrame,
    expected: List[str],
    reason: str,
):
    """
    Checks that the get_features function is working properly.
    """
    assert train_test_mock_model.get_features(data) == expected, reason
