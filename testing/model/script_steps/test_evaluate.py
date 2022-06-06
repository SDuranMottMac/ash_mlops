"""
Contributors: Antoine Chammas

Summary: This file contains basic functions for testing evaluation scripts.
NOTE: Parameters could be added to the testing samples to save space in this file.
"""

# could-change this file to implement the tests you need.

from typing import Dict, List, Union

from model.script_steps import evaluate as evaluate
from testing import testing_helpers as th
from testing.model.script_steps.test_samples import testing_samples_evaluate as ts

from requests import Response
import pytest
import pandas as pd


@pytest.fixture(autouse=True)
def mock_arg_parse(mocker):
    mocker.patch.object(evaluate.Evaluate, "get_model_endpoint", return_value="")


@pytest.fixture(autouse=True)
def simple_evaluator():
    return evaluate.Evaluate()


@pytest.mark.skipif(
    not th.check_func_in_obj("get_existing_model_predictions", evaluate.Evaluate),
    reason="No function called get_existing_model_predictions",
)
@pytest.mark.parametrize(
    "expected,reason,status_code,model_preds,error_case",
    [
        (
            Exception(),
            "get_existing_model_predictions response status code error raising"
            + " not working properly",
            404,
            None,
            True,
        ),
        (
            Exception("Predictions not returned as array."),
            "get_existing_model_predictions predictions not returned as array"
            + " not working properly",
            200,
            {},
            True,
        ),
        (
            Exception("Number of rows mismatch: len(data) != len(predictions)."),
            "get_existing_model_predictions predictions returned as array"
            + " but with less length than data not working properly",
            200,
            [1],
            True,
        ),
        (
            Exception("Number of rows mismatch: len(data) != len(predictions)."),
            "get_existing_model_predictions predictions returned as array"
            + " but with more length than data not working properly",
            200,
            [1, 2, 3, 4, 5],
            True,
        ),
        (
            [1, 2, 3],
            "get_existing_model_predictions predictions returned as array"
            + " but with same length as data not working properly",
            200,
            [1, 2, 3],
            False,
        ),
    ],
)
def test_get_existing_model_predictions(
    mocker,
    simple_evaluator: evaluate.Evaluate,
    expected: List[Union[int, float]],
    reason: str,
    status_code: int,
    model_preds: Union[List[Union[int, float]], None, Dict],
    error_case: bool,
):
    """
    Checks that the get_existing_model_predictions function is working properly.

    Args:
        mocker ([type]): [description]
        expected (List[Union[int, float]]): [description]
        reason (str): [description]
        status_code (int): [description]
        model_preds (Union[List[Union[int, float]], None, Dict]): [description]
    """
    mocked_response = Response()
    mocked_response.status_code = status_code
    mocker.patch("requests.post", return_value=mocked_response)
    mocker.patch.object(Response, "json", return_value=model_preds)

    if error_case:
        with pytest.raises(Exception) as exception:
            simple_evaluator.get_existing_model_predictions(pd.DataFrame())
        assert (
            type(exception.value) is type(expected)
            and exception.value.args == expected.args
        ), reason
    else:
        assert (
            simple_evaluator.get_existing_model_predictions(
                pd.DataFrame(ts.bidh_ensemble_model_evaluation_dataframe_0)
            )
            == expected
        ), reason


@pytest.mark.skipif(
    not th.check_func_in_obj("sum_rain_cols", evaluate.Evaluate),
    reason="No function called sum_rain_cols",
)
@pytest.mark.parametrize(
    "data,expected,reason",
    [
        (
            pd.DataFrame(dtype="float64"),
            pd.Series(dtype="float64"),
            "sum_rain_cols should return an empty dataframe in this case",
        ),
        (
            pd.DataFrame({"rain_col": [1, 2, 3, 4]}, dtype="float64"),
            pd.Series([0, 0, 0, 0], dtype="float64"),
            "sum_rain_cols should return zero-ed dataframe in this case",
        ),
        (
            pd.DataFrame(
                {"rain_window_1": [1, 2, 3, 4], "rain_window_2": [4, 3, 2, 1]},
                dtype="float64",
            ),
            pd.Series([5, 5, 5, 5], dtype="float64"),
            "sum_rain_cols should return a summed dataframe in this case",
        ),
        (
            pd.DataFrame(
                {
                    "rain_window_1": [1, 2, 3, 4],
                    "rain_window_b": [4, 3, 2, 1],
                    "rain_window_ 4": [4, 3, 2, 1],
                    "rain_col": [1, 2, 3, 4],
                },
                dtype="float64",
            ),
            pd.Series([5, 5, 5, 5], dtype="float64"),
            "sum_rain_cols should return a summed dataframe in this case"
            + "and ignore the irrelevant columns.",
        ),
    ],
)
def test_sum_rain_cols(
    simple_evaluator: evaluate.Evaluate,
    data: pd.DataFrame,
    expected: pd.Series,
    reason: str,
):
    """
    Checks that the sum_rain_cols function is working properly.

    Args:
        data (pd.DataFrame): [description]
        expected (pd.Series): [description]
        reason (str): [description]
    """
    assert simple_evaluator.sum_rain_cols(data).equals(expected), reason


@pytest.mark.skipif(
    not th.check_func_in_obj("filter_rs_top_data", evaluate.Evaluate),
    reason="No function called filter_rs_top_data",
)
@pytest.mark.parametrize(
    "data,percentile,expected,reason",
    [
        (
            pd.DataFrame({"rain_summation": list(range(1, 101))}, dtype="float64"),
            0.5,
            pd.DataFrame({"rain_summation": list(range(51, 101))}, dtype="float64"),
            "filter_rs_top_data is not working properly for a 0.5 percentile.",
        ),
        (
            pd.DataFrame({"rain_summation": list(range(1, 101))}, dtype="float64"),
            0.97,
            pd.DataFrame({"rain_summation": list(range(98, 101))}, dtype="float64"),
            "filter_rs_top_data is not working properly for a 0.97 percentile.",
        ),
    ],
)
def test_filter_rs_top_data(
    simple_evaluator: evaluate.Evaluate,
    data: pd.DataFrame,
    percentile: float,
    expected: pd.DataFrame,
    reason: str,
):
    """
    Checks that the filter_rs_top_data function is working properly.
    Needs a rain_summation column to function.

    Args:
        data (pd.DataFrame): [description]
        percentile (float): [description]
        expected (pd.Series): [description]
        reason (str): [description]
    """
    assert all(
        simple_evaluator.filter_rs_top_data(data, percentile=percentile).values
        == expected.values
    ), reason


@pytest.mark.skipif(
    not th.check_func_in_obj("summarise_and_score", evaluate.Evaluate),
    reason="No function called summarise_and_score",
)
@pytest.mark.parametrize(
    "data,expected,reason,error_case",
    [
        (
            ts.bidh_ensemble_model_sumnscore_input_datasample_0,
            ts.bidh_ensemble_model_sumnscore_output_datasample_0,
            "null case isn't erroring out as expected",
            True,
        ),
        (
            ts.bidh_ensemble_model_sumnscore_input_datasample_1,
            ts.bidh_ensemble_model_sumnscore_output_datasample_1,
            "partial null case isn't erroring out as expected",
            True,
        ),
        (
            ts.bidh_ensemble_model_sumnscore_input_datasample_2,
            ts.bidh_ensemble_model_sumnscore_output_datasample_2,
            "all zeros case isn't working as expected",
            False,
        ),
        (
            ts.bidh_ensemble_model_sumnscore_input_datasample_3,
            ts.bidh_ensemble_model_sumnscore_output_datasample_3,
            "existing model being better case isn't working as expected",
            False,
        ),
        (
            ts.bidh_ensemble_model_sumnscore_input_datasample_4,
            ts.bidh_ensemble_model_sumnscore_output_datasample_4,
            "training model being better case isn't working as expected",
            False,
        ),
    ],
)
def test_summarise_and_score(
    simple_evaluator: evaluate.Evaluate,
    data: pd.DataFrame,
    expected: Dict[str, Union[float, int, bool, List[Union[float, int]]]],
    reason: str,
    error_case: bool,
):
    """
    Checks that the summarise_and_score function is working properly.
    NOTE: This function doesn't work properly if given None data and will raise
    value errors, make sure that this is the desired functionality
    """
    if error_case:
        with pytest.raises(Exception) as exception:
            simple_evaluator.summarise_and_score(data)
        assert (
            type(exception.value) is type(expected)
            and exception.value.args == expected.args
        ), reason
    else:
        assert simple_evaluator.summarise_and_score(data) == expected, reason
