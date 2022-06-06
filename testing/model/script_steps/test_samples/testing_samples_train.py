"""
Contributors: Antoine Chammas

Summary: This file contains objects or values that can be used as samples
for testing purposes - Train Samples File
"""

import pandas as pd

#############################
# Bangkok-IDH Traffic Model #
#############################

bidh_ensemble_model_hyperparams_sample_0 = {
    "xgb_model_n_estimators": 100,
    "xgb_model_reg_lambda": 1,
    "xgb_model_gamma": 0,
    "xgb_model_max_depth": 64,
    "arima_model_order": [2, 1, 2],
    "arima_model_seasonal_order": [0, 0, 1, 8],
    "arima_model_max_iter": 250,
}

bidh_ensemble_model_invalid_hyperparams_sample_0 = {
    "xgb_model_n_estimators": "100",
    "xgb_model_reg_lambda": 1,
    "xgb_model_gamma": 0,
    "xgb_model_max_depth": "64",
    "arima_model_order": [2, 1, 2],
    "arima_model_seasonal_order": [0, 0, 1, 8],
    "arima_model_max_iter": 250,
}

bidh_ensemble_model_invalid_hyperparams_sample_1 = {
    "xgb_model_n_estimators": 100,
    "xgb_model_reg_lambda": 1,
    "xgb_model_gamma": 0,
    "arima_model_order": [2, 1, 2],
    "arima_model_seasonal_order": [0, 0, 1, 8],
    "arima_model_max_iter": 250,
}

bidh_ensemble_model_arima_mse_0 = [1, 1, 1, 1]

bidh_ensemble_model_xgb_mse_0 = [1, 1, 1, 1]

bidh_ensemble_model_prediction_0 = [1, 1, 1, 1]

bidh_ensemble_model_stats_0 = {
    "arima_in_sample_mse": bidh_ensemble_model_arima_mse_0,
    "xgb_in_sample_mse": bidh_ensemble_model_xgb_mse_0,
    "combined_in_sample_mse": 0,
}

bidh_ensemble_model_calccoeffs_0 = {"xgb": 0.12800904, "arima": 0.87219783}

bidh_ensemble_model_train_data_cols_0 = pd.DataFrame(
    columns=["rain_window_0", "rain_window_1", "rain_window_2"]
)

bidh_ensemble_model_train_data_cols_1 = pd.DataFrame(
    columns=list(bidh_ensemble_model_train_data_cols_0.columns)
    + ["test_rain_window_0", "rein_window_1", "RAIN_WINDOW_1"]
)

bidh_ensemble_model_features_0 = ["rain_window_0", "rain_window_1", "rain_window_2"]
