"""
Contributors: Antoine Chammas

Summary: This file contains objects or values that can be used as samples
for testing purposes - Evaluate Samples File
"""

import pandas as pd

#############################
# Bangkok-IDH Traffic Model #
#############################

bidh_ensemble_model_predictions_sample_0 = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]

bidh_ensemble_model_predictions_sample_1 = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75]

bidh_ensemble_model_predictions_sample_2 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

bidh_ensemble_model_predictions_sample_3 = [0, 0, 0, 0, 0, 0]

bidh_ensemble_model_predictions_sample_4 = [None, None, None, None, None, None]

# TODO: rename this to a less general name
bidh_ensemble_model_evaluation_dataframe_0 = pd.DataFrame([1, 2, 3])

# case where all the data is null
bidh_ensemble_model_sumnscore_input_datasample_0 = pd.DataFrame(
    {
        "training_prediction": bidh_ensemble_model_predictions_sample_4,
        "gauge_level": bidh_ensemble_model_predictions_sample_4,
        "existing_prediction": bidh_ensemble_model_predictions_sample_4,
    }
)

bidh_ensemble_model_sumnscore_output_datasample_0 = ValueError(
    "Input contains NaN, infinity or" + " a value too large for dtype('fl" + "oat64')."
)

# case where part of the data is null
bidh_ensemble_model_sumnscore_input_datasample_1 = pd.DataFrame(
    {
        "training_prediction": bidh_ensemble_model_predictions_sample_0,
        "gauge_level": bidh_ensemble_model_predictions_sample_4,
        "existing_prediction": bidh_ensemble_model_predictions_sample_1,
    }
)

bidh_ensemble_model_sumnscore_output_datasample_1 = ValueError(
    "Input contains NaN, infinity or" + " a value too large for dtype('fl" + "oat64')."
)

# case where we have all 0s
bidh_ensemble_model_sumnscore_input_datasample_2 = pd.DataFrame(
    {
        "training_prediction": bidh_ensemble_model_predictions_sample_3,
        "gauge_level": bidh_ensemble_model_predictions_sample_3,
        "existing_prediction": bidh_ensemble_model_predictions_sample_3,
    }
)

bidh_ensemble_model_sumnscore_output_datasample_2 = {
    "register_trained_model": False,
    "training_mse_rain_events": 0.0,
    "existing_mse_rain_events": 0.0,
    "training_mse_whole_sample": 0.0,
    "existing_mse_whole_sample": 0.0,
}

# case where existing model is better
bidh_ensemble_model_sumnscore_input_datasample_3 = pd.DataFrame(
    {
        "training_prediction": bidh_ensemble_model_predictions_sample_0,
        "gauge_level": bidh_ensemble_model_predictions_sample_1,
        "existing_prediction": bidh_ensemble_model_predictions_sample_2,
    }
)

bidh_ensemble_model_sumnscore_output_datasample_3 = {
    "register_trained_model": False,
    "training_mse_rain_events": 0.25,
    "existing_mse_rain_events": 0.0625,
    "training_mse_whole_sample": 0.25,
    "existing_mse_whole_sample": 0.0625,
}

# case where training model is better
bidh_ensemble_model_sumnscore_input_datasample_4 = pd.DataFrame(
    {
        "training_prediction": bidh_ensemble_model_predictions_sample_2,
        "gauge_level": bidh_ensemble_model_predictions_sample_1,
        "existing_prediction": bidh_ensemble_model_predictions_sample_0,
    }
)

bidh_ensemble_model_sumnscore_output_datasample_4 = {
    "register_trained_model": True,
    "training_mse_rain_events": 0.0625,
    "existing_mse_rain_events": 0.25,
    "training_mse_whole_sample": 0.0625,
    "existing_mse_whole_sample": 0.25,
}
