"""
This script step is where we initialise and train our model.

This is a sample taken from a project utilising both `shared_code` and an
AML File Dataset.

The pre-processing code found here could also be written under its own
'preprocessing' step.

# should-change refer to the README doc, you should have a script named
train.py and you can use either this file or train_tabular as your
skeleton for that file.

"""
from pathlib import Path
import tempfile
import json
import os
from azureml.core.run import Run
import pandas as pd
import numpy as np

from shared_code.ml_utils.model_builder import build_sequential_model, quantile_loss
from shared_code.ml_utils.model_trainer import train_sequential_model
from shared_code.ml_utils.normalizer import Normalizer
from shared_code.overflow_prediction.data_cleaner import clean_data_optimised
from shared_code.overflow_prediction.feature_generator import (
    generate_rain_flow_lagged_features,
)


run = Run.get_context()

if __name__ == "__main__":
    # load training and testing datasets
    file_dataset = run.input_datasets["train_data"]
    input_path = os.path.dirname(__file__)
    download_path = os.path.dirname(
        file_dataset.download(target_path=input_path, overwrite=False)[0]
    )

    raw_data = pd.read_csv(
        f"{download_path}/raw_data.csv", index_col="time", parse_dates=True
    )
    gauges = json.load(open(f"{download_path}/gauge_summaries.json", "r"))

    rain_gauges = [str(gauge) for gauge in gauges["rainfall"]]
    flow_gauges = [str(gauge) for gauge in gauges["flow"]]

    RESAMPLING_FREQUENCY = "5T"

    cleaned_data = clean_data_optimised(
        raw_data,
        rain_gauges=rain_gauges,
        flow_gauges=flow_gauges,
        resampling_frequency=RESAMPLING_FREQUENCY,
    )

    cleaned_data_features = generate_rain_flow_lagged_features(
        cleaned_data, rain_gauges=rain_gauges, flow_gauges=flow_gauges
    )

    cleaned_data_and_features = pd.concat(
        [cleaned_data_features, cleaned_data], axis=1
    ).dropna()

    TRAINING_START = "2020-01-01 00:00:00"
    TRAINING_END = "2021-01-01 00:00:00"

    training_data = cleaned_data_and_features[TRAINING_START:TRAINING_END].reset_index(
        drop=True
    )

    y_train = training_data[flow_gauges]
    x_train = training_data[
        [col for col in training_data.columns if col not in flow_gauges]
    ]

    x_train_np = np.asarray(x_train)
    y_train_np = np.asarray(y_train)

    scaler = Normalizer().fit(x_train_np)
    x_train_scaled = scaler.transform(x_train_np)

    input_shape = x_train.shape[1]

    model = build_sequential_model(
        model_name="mlp_overflow_prediction",
        input_dimensions=input_shape,
        nodes_list=[128, 128, 64, 32],
        model_options={
            "n_outputs": 4,
            "loss": lambda y, f: quantile_loss(0.5, y, f),
            "metrics": ["mse", "mae", "mape"],
            "reg_options": {"l2": 0.01},
            "optimizer_options": {
                "learning_rate": 0.001,
                "beta_1": 0.9,
                "beta_2": 0.999,
            },
            "random_seed": 123,
        },
    )

    model, history = train_sequential_model(
        model=model,
        x_train=x_train_scaled,
        y_train=y_train_np,
        fit_options={
            "validation_split": 0.2,
        },
        early_stop_monitor="val_loss",
    )

    model_file_path = Path(run.output_datasets["model"])
    temp = tempfile.mkdtemp()
    model.save(temp / Path("tensorflow_saved_model"))
    for path in [f.relative_to(temp) for f in Path(temp).rglob("*") if f.is_file()]:
        (model_file_path / path).parent.mkdir(parents=True, exist_ok=True)
        with open(temp / path, "rb") as source:
            with open(model_file_path / path, "wb") as destination:
                destination.write(source.read())

    # save history
    with open((model_file_path / Path("history.json")), "w", encoding="utf8") as file:
        json.dump(history.history, file)
