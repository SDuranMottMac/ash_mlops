"""
    Contributors: Evan Harwin, Antoine Chammas

    Summary: Create a Model Training Pipeline that can be used in our MLOps Practice

    See the docs:
    https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines
    https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/machine-learning-pipelines/intro-to-pipelines/aml-pipelines-getting-started.ipynb

    Detailed Description:
    This file outlines a simple three-step pipeline, that should cover most model deployments.
    `train -> evaluate -> register`

    The order of operation in this file is something like the following:

        Run `create_and_run_pipeline`:

            1. First, connects to the workspace with `connect_and_get_workspace`.
                - Using cli authentication and `get_workspace`

            2. Then creates the pipeline with `create_pipeline`.

                - This initially fetches the AML dependencies:
                  - a datastore (we just use the default)
                  - a compute cluster (using compute_name)
                  - an environment (using environment var `ML_ENVIRONMENT_NAME`)
                  - a runconfig (only dependant on the environment)

                - Then we use `get_train_and_test_data` to get the train and test datasets from our
                  datastore.

                - Next, we define our script step outputs `model` and `model_score`

                - Finally we use `declare_steps` to define the script steps (`train`, `evaluate` and
                  `register`)

            3. Runs the pipeline using `run_pipeline`

"""

import os
from azureml.core import Workspace, Dataset, Experiment, Environment, RunConfiguration
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.data import OutputFileDatasetConfig
from azureml.core.authentication import AzureCliAuthentication


class PipelineRunner:
    """Pipeline Runner Class."""

    def __init__(self):
        """
        Creates Pipeline Runner
        """
        self.compute = None
        self.runconfig = None
        self.train_data = None
        self.test_data = None
        self.model = None
        self.model_score = None

    def create_and_run_pipeline(self):
        """Get Workspace, Create Pipeline, Run Pipeline."""
        workspace = self.connect_and_get_workspace()
        pipeline = self.create_pipeline(workspace)
        self.run_pipeline(pipeline, workspace, wait_for_completion=True)

    def create_pipeline(self, workspace):
        """Fetches Dependencies for the AML Pipeline:
        - datastore (using `workspace.get_default_datastore`)
        - compute (using `compute_name`)
        - environment (using os.environ.get('ML_ENVIRONMENT_NAME')
          and the conda specification)
        - runconfig (just uses the environment)

        Gets the input datasets (using `get_train_and_test_data`):
        - train
        - test

        Defines the output datasets:
        - model
        - model_score

        Declares the pipeline steps using `declare_steps`
        and returns an AML Pipeline object.
        """
        datastore = workspace.get_default_datastore()
        self.compute = workspace.compute_targets[os.environ.get("ML_TRAINING_COMPUTE")]
        environment = Environment.from_conda_specification(
            os.environ.get("ML_ENVIRONMENT_NAME"),
            f"./{os.environ.get('ML_PATH')}/config/conda_environment.yml",
        )

        # could-change
        # sample if you need to download a private pip package
        # whl_url = Environment.add_private_pip_wheel(
        #     workspace=workspace,
        #     file_path="./wheels/yourwheel.whl",
        #     exist_ok=True,
        # )
        # environment.python.conda_dependencies.add_pip_package(whl_url)

        self.runconfig = RunConfiguration()
        self.runconfig.environment = environment

        # get training and testing data
        self.train_data, self.test_data = self.get_train_and_test_data(workspace)

        # initialize output configuration
        self.model = OutputFileDatasetConfig(
            "model", (datastore, "/model-in-training/{run-id}/{output-name}/")
        )
        self.model_score = OutputFileDatasetConfig(
            "model_score", (datastore, "/model-in-training/{run-id}/{output-name}/")
        )

        # declare the pipeline steps
        steps = self.declare_steps(step_order=["train", "evaluate", "register"])

        # instantiate the pipeline and return
        return Pipeline(workspace=workspace, steps=steps)

    @staticmethod
    def get_train_and_test_data(workspace):
        """
        We load two preprepared datasets from the given datastore,
        one for testing and one for training.
        The paths to these files are stored in the environment under:
        `ML_TRAIN_DATA_PATH` and `ML_TEST_DATA_PATH`.

        These files' format has to be what is expected by
        the training and evaluation scripts.
        """
        train_data_path = os.environ.get("ML_TRAIN_DATA_PATH")
        test_data_path = os.environ.get("ML_TEST_DATA_PATH")
        train_data = Dataset.get_by_name(workspace, name=train_data_path)
        test_data = Dataset.get_by_name(workspace, name=test_data_path)
        return train_data, test_data

    def declare_steps(self, step_order, source_directory="./"):
        """
        Here we configure our script steps, one for each file in the `/model/script_steps/`
        directory.

        These are largely configured the same here, using the same `compute` and
        `runconfig`.

        However the inputs and outputs are different datasets,
        defined in `get_train_and_test_data`
        and `create_pipeline`.
        """
        if "train" in step_order:
            train_step = PythonScriptStep(
                name="train",
                source_directory=source_directory,
                script_name=f"{os.environ.get('ML_PATH')}/script_steps/train.py",
                inputs=[self.train_data.as_named_input("train_data")],
                outputs=[self.model.as_mount()],
                compute_target=self.compute,
                runconfig=self.runconfig,
            )

        if "evaluate" in step_order:
            evaluate_step = PythonScriptStep(
                name="evaluate_registered_model",
                source_directory=source_directory,
                script_name=f"{os.environ.get('ML_PATH')}/script_steps/evaluate.py",
                inputs=[
                    self.model.as_input("model").as_download(),
                    self.test_data.as_named_input("test_data"),
                ],
                outputs=[self.model_score.as_mount()],
                arguments=["--model-endpoint", os.environ.get("ML_DEPLOYED_ENDPOINT")],
                compute_target=self.compute,
                runconfig=self.runconfig,
            )

        if "register" in step_order:
            register_step = PythonScriptStep(
                name="register_model",
                source_directory=source_directory,
                script_name=f"{os.environ.get('ML_PATH')}/script_steps/register.py",
                inputs=[
                    self.model.as_input("model").as_download(),
                    self.model_score.as_input("model_score").as_download(),
                ],
                compute_target=self.compute,
                runconfig=self.runconfig,
            )

        store = {
            "train": train_step,
            "evaluate": evaluate_step,
            "register": register_step,
        }

        steps = []

        for step in step_order:
            if step not in store:
                raise Exception(
                    str(
                        "Couldn't assign step, Error: Step "
                        + str(step)
                        + "not supported."
                    )
                )
            steps.append(store[step])
        return steps

    @staticmethod
    def run_pipeline(pipeline, workspace, wait_for_completion=False):
        """Run the Pipeline - fingers crossed!

        If you wish to trigger the pipeline from another script,
        you could also register it here.
        This gives it a REST endpoint.
        """
        experiment = Experiment(workspace, "ModelTrainingPipeline")
        pipeline_run = experiment.submit(pipeline)
        if wait_for_completion:
            pipeline_run.wait_for_completion()

    def connect_and_get_workspace(self):
        """
        Connect to Workspace using environment variables:
        - `ML_WORKSPACE_NAME`
        - `ML_RESOURCE_GROUP`
        - `ML_SUBSCRIPTION_ID`
        """
        cli_auth = AzureCliAuthentication()

        workspace_name = os.environ.get("ML_WORKSPACE_NAME")
        resource_group = os.environ.get("ML_RESOURCE_GROUP")
        subscription_id = os.environ.get("ML_SUBSCRIPTION_ID")

        return self.get_workspace(
            workspace_name, subscription_id, resource_group, cli_auth
        )

    @staticmethod
    def get_workspace(name, subscription_id, resource_group, auth=None):
        """
        Function used to get an azure ml workspace.

        Parameters:
        name (String): Name of the workspace you want to get.
        subscription_id (String): Name of the subscription id
        that has access to the workspace you want to get.
        resource_group (String): Name of the resource group
        that the workspace is in.
        auth (ServicePrincipalAuthentication/AzureCliAuthentication (Pipeline)
                or InteractiveLoginAuthentication (Local)):
            Authentication used for the workspace
            defaults to None. If set to None, this will cause the tests to 'hang'
            in the pipeline.

        Useful Links:
        https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.workspace.workspace?view=azure-ml-py
        https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.authentication.serviceprincipalauthentication?view=azure-ml-py
        https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.authentication.interactiveloginauthentication?view=azure-ml-py

        Returns:
        Returns the model.
        If an Exception occurs:
            - The exception as a string.

        TODO:
        - Refactor this to be similar to the env variable in terms of
        appending errors.
        """
        workspace = Workspace.get(
            name=name,
            subscription_id=subscription_id,
            resource_group=resource_group,
            auth=auth,
        )
        return workspace


# running this script creates and runs the pipeline
# on aml compute
if __name__ == "__main__":
    pipeline_runner = PipelineRunner()
    pipeline_runner.create_and_run_pipeline()
