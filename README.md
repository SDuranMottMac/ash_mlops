# AML Studio MLOps Model Standard

- [AML Studio MLOps Model Standard](#aml-studio-mlops-model-standard)
  - [Naming Conventions](#naming-conventions)
  - [Development Notes](#development-notes)
  - [So you have a new model](#so-you-have-a-new-model)
    - [Steps](#steps)
      - [1. Clone this repo OR duplicate a previously existing model folder (for another model in this repo)](#1-clone-this-repo-or-duplicate-a-previously-existing-model-folder-for-another-model-in-this-repo)
      - [2. Make a new DevOps CI pipeline using the `model/devops_pipeline.yml`](#2-make-a-new-devops-ci-pipeline-using-the-modeldevops_pipelineyml)
      - [3. Check the default AML pipeline suits your needs](#3-check-the-default-aml-pipeline-suits-your-needs)
      - [4. Add your model to the `model/script_steps/train.py` script](#4-add-your-model-to-the-modelscriptstepstrainpy-script)
      - [5. Add your evaluation steps into the `model/script_steps/evaluate.py` script](#5-add-your-evaluation-steps-into-the-modelscriptstepsevaluatepy-script)
      - [6. Update the model tags, name and description in the `model/script_steps/register.py` script](#6-update-the-model-tags-name-and-description-in-the-modelscriptstepsregisterpy-script)
      - [7. Update the model/config/entry_script.py to match the data requirements for your model](#7-update-the-modelconfigentry_scriptpy-to-match-the-data-requirements-for-your-model)
      - [8. Check the `model/config/deployment_config_dev.yml` and `model/config/deployment_config_prd.yml`](#8-check-the-modelconfigdeployment_config_devyml-and-modelconfigdeployment_config_prdyml)
      - [9. Update the `model/config/conda_environment.yml` for your model](#9-update-the-modelconfigconda_environmentyml-for-your-model)
      - [10. Add any needed unit tests of the above Python scripts to   `model/tests/`](#10-add-any-needed-unit-tests-of-the-above-python-scripts-to---modeltests)
      - [11. Add CD Release Pipeline to DevOps](#11-add-cd-release-pipeline-to-devops)
  - [Using the AML Pipeline](#using-the-aml-pipeline)
    - [Registering the training and testing datasets](#registering-the-training-and-testing-datasets)
    - [Moving data between Python script steps](#moving-data-between-python-script-steps)
      - [Inputs](#inputs)
      - [Outputs](#outputs)
      - [Outputs as Inputs](#outputs-as-inputs)
  - [Writing Unit Tests](#writing-unit-tests)
    - [Tests Location](#tests-location)
    - [Using Pytest](#using-pytest)
      - [Using Pytest Fixtures](#using-pytest-fixtures)
      - [Using Pytest Mocker](#using-pytest-mocker)
      - [Using Pytest Skipif](#using-pytest-skipif)
      - [Using Pytest Parametrize](#using-pytest-paramterize)
      - [Pytest.ini File](#pytest.ini-file)
    - [Writing a Unit Test](#writing-a-unit-test)
  - [Using the MLOps Folder Structure](#using-the-mlops-folder-structure)
    - [Models](#models)
    - [Shared Code](#shared-code)
    - [Testing](#testing)
    - [VirtualEnvs](#virtualenvs)

## Naming Conventions

For naming your personal repositories and teams related to that repository please refer to [this document](https://github.com/orgs/mottmacdonaldglobal/teams/data-science-community/discussions/1) about naming conventions.

## Development Notes

TODO:

## So you have a new model

### Steps

#### 1. Clone this repo OR duplicate a previously existing model folder (for another model in this repo)

You probably want to change the directory name `model/` to the going name of your model, something like: `canal-level-forecasting/`

Cloning the repository, rather than having multiple models per repository is advised in most cases.

**⚠️ Most of the files not mentioned in this template can stay as is but certain elements of them might have to be changed. These elements will have `should-change` (probably mandatory) or `could-change` (most likely unnecessary but could) tags next to them. Some elements that had to be changed might not have to be changed anymore after an update.**

#### 2. Make a new DevOps CI pipeline using the `devops_pipeline.yml`

You will likely need to make some changes to the file, just the  variables at the top of the config need to match your chosen AML workspace and the model itself.

The artifact for the pipeline will be this repository, either you have cloned it or duplicated the model folder.

The pipeline will be specified by `devops_pipeline.yml`.

After this is done; you need to setup a new pipeline in the DevOps interface for this model, using the aforementioned yaml.

This is an example of how your pipeline might look on DevOps:
![image](https://user-images.githubusercontent.com/20205739/162221981-07f6560c-a685-436a-8e45-ad398e0a4037.png)

#### 3. Check the default AML pipeline suits your needs

Just a quick glance over the file is wise, you might want to change:

- The experiment name in `run_pipeline`
- The script steps used in `declare_steps` and their `script_name`.

Be sure to read over `declare_steps` as that function defines the script steps, which are the core of our AML pipeline.

We currently use a procedure as follows:

```train -> evaluate -> register```

However, perhaps you want a structure more like one of the following:

```optimise_hyperparams -> train -> evaluate -> register```

...or if your data needs to be preprocessed prior to training somehow you might want to tack a data preparation step onto the front:

```process_data -> train -> evaluate -> register```

All of these will require more significant rewrites to this repo however, so consider the value that they provide carefully before committing to these changes!

**⚠️Important Note: In the script steps examples we provide different training scripts based on the type of Azure Machine Learning datasets you would be using (File/Tabular). Please either rename the one that fits your needs best to `train.py` or create a `train.py` file from scratch which is the naming used in this document.**

#### 4. Add your model to the `model/script_steps/train.py` script

_In most cases, this is probably the most involved step._

For a simple model, imported from sklearn, XGBoost or statsmodels this could be as easy as:

1. defining model w/ hyperparams
2. training model on given dataset
3. saving model as pickle file

However, we can also go big here with:

- complex hyperparameter optimisation
- custom models
- in-depth training regimes

...anything is possible, just: training dataset in, model out!
  
#### 5. Add your evaluation steps into the `model/script_steps/evaluate.py` script

This is a good opportunity to make sure that your model is performing exactly as it should. Add tests and validation, data specific scores and whatever metrics make most sense for the context of your model.

The chances are you know exactly what you want your model to be good at, so make sure you add that here.

Things to consider may be:

- The importance of a low false negative rate
- The model accuracy at specific events that we can filter out of the dataset
- If the models accuracy matters in historic data

#### 6. Update the model tags, name and description in the `model/script_steps/register.py` script

As simple as it sounds, as long as your evaluation script is saving `summary.json` file with the boolean property `register_trained_model`, then very little needs to change here.

#### 7. Update the model/config/entry_script.py to match the data requirements for your model

Just glance over the entry script and check that the `init` function will load your model, and the `run` function will return model predictions.

#### 8. Check the `model/config/deployment_config_dev.yml` and `model/config/deployment_config_prd.yml`

These should be pretty standard, but just check that they are setup to meet your models compute requirements. If you are unsure which compute you should use, please refer to this [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-target#deploy).

#### 9. Update the `model/config/conda_environment.yml` for your model

Make sure that any Python requirement that your model has is added.

Also check now that these requirements are imported in `evaluate.py`, `train.py` and `entry_script.py`.

#### 10. Add any needed unit tests of the above Python scripts to   `model/tests/`

This doesn't need to be 100% coverage - before finalization - but now is a good time to read through all of the scripts you have just changed and see if any suspect lines need to be put into functions and unit tested. We don't want to be deploying models with off-by-one errors!

#### 11. Add CD Release Pipeline to DevOps

This stage we have to do by hand using the DevOps Web GUI.

You should be able to exactly replicate this one:
<https://dev.azure.com/mmdigitalventures/Moata/_release?view=all&_a=releases&definitionId=89>

Read through the tasks, and change the name of the model and the Azure ML Workspace as required.

## Using the AML Pipeline

### Registering the training and testing datasets

In order to register your training and testing datasets you need to open your [Azure Machine Learning Studio](https://studio.azureml.net/) instance then click on `Datasets` and then `Create Dataset`. The source of your dataset can be local or online, it depends on what data you are using.

Next, you will have to give your dataset a name, a type and a description.

- The dataset name should follow the following naming convention: `modeltype-project-usage-version`, for example `canallevelforecast-bangkok-validation-01` or `demandforecast-auckland-general-04`.
- If you are providing only a tabular file i.e. a csv file, then use the `Tabular` type, if you are providing multiple files or a non-tabular file, then use the `File` type.
- For the description, this is what recommend to include:
  - Data source(s) - Where the data is collected from.
  - Data content - What the data contains.
  - Original purpose - Your original usage of the data, i.e. the task you are using it for.
  - Contacts - Email of the owner(s) of the dataset.

Depending on the type of dataset you choose, you will be met with different options. This [documentation][https://docs.microsoft.com/en-us/azure/machine-learning/how-to-connect-data-ui?tabs=credential#create-datasets] can help you set everything up.

It is recommended to also add tags to your dataset in order to filter them if needs be, sample tags can be `Usage` or `Models Using This Dataset`.

**⚠️Important Note: If using training data locally, it should be stored under the `data` folder in your `model` folder. The `data` folder should always be git ignored. You should never share data through Github. All raw data should be stored in a secure commonly accessible location and interim data (i.e. cleaned raw data, training and testing data sets) should be created through the code and kept in this folder locally.**

### Moving data between Python script steps

#### Inputs

Firstly, to train our model (or perhaps you might be feeding the input dataset into data preparation
script steps) we will need to ingest a dataset.

We will use datasets stored and registered in AML Studio, and these can be dynamic or static.

There are two different types of AML Dataset we will use: Files and Tabular. In this example we load our data using the Tabular Dataset `from_delimited_files` method, like so:

```python
# in aml_pipeline.py
train_data = Dataset.Tabular.from_delimited_files(
    path=[(datastore, train_data_path)]
)
```

We uploaded our data to the datastore earlier, using the method described in this document.

Then we can pass this to the model by adding it to `inputs` when we declare steps:

```python
# in aml_pipeline.py
train_step = PythonScriptStep(
    name="train",
    source_directory=source_directory + 'train',
    script_name='train.py',
    inputs=[
        train_data.as_named_input('train_data')
    ],
    outputs=[
        model.as_mount()
    ],
    compute_target=compute,
    runconfig=runconfig
)
```

Then in the Python script file `train.py` we load it from the run context like so:

```python
# in train.py
from azureml.core.run import Run
run = Run.get_context()
train = run.input_datasets['train_data'].to_pandas_dataframe()
```

We can also pass data in as arguments to the script step, and fetch it using `sys.argv[n]`, but the run context is cleaner.

For simplicity, in the current `aml_pipeline.py` file, datasets are loaded agnostically by name. You can use the different `File` or `Tabular` functionalities inside of your script steps.

#### Outputs

We also often want to output data from a script step. This may include a `model.pkl` or a scoring file that summarises your models performance.

To do this we need to define an `OutputFileDatasetConfig`. This looks like so:

```python
# in the aml_pipeline.py
model = OutputFileDatasetConfig(
    "model",
    (datastore, "/model-in-training/{run-id}/{output-name}/")
)
```

Then, we add it to the script step outputs `as_mount`. Again, if we wish we can pass it in as an argument, but the run context is neater.

```python
# in aml_pipeline.py
train_step = PythonScriptStep(
    name="train",
    source_directory=source_directory + 'train',
    script_name='train.py',
    inputs=[
        train_data.as_named_input('train_data')
    ],
    outputs=[
        model.as_mount()
    ],
    compute_target=compute,
    runconfig=runconfig
)
```

Saving files to the output in the train script can be a little fiddly, but if you follow a process similar to this you should be fine:

```python
# in train.py
import pickle
from azureml.core.run import Run

run = Run.get_context()

# run.output_datasets['dataset_name'] returns a string path
# when dataset output passed in as_mount
model_file_path = run.output_datasets['model']

with open(os.path.join(model_file_path, 'model.pkl'), 'wb') as file:
    pickle.dump(model_instance, file)
```

#### Outputs as Inputs

The final case to document, sometimes we will want to feed the outputs from a previous script step into the current.

Firstly you will define an `OutputFileDatasetConfig` as above, then use it as an output in a script step (we will assume that this has been done exactly as above with the `model` output).

Then, to feed it into the next script step as an input you add it to the script step inputs in the `aml_pipeline.py` as so:

```python
# in aml_pipeline.py
evaluate_step = PythonScriptStep(
    name="evaluate_registered_model",
    source_directory=source_directory + 'evaluate',
    script_name='evaluate.py',
    inputs=[
        model.as_input('model').as_download(), # this line, where the model output from above is used
                                               # as an input into the evaluate script step
        test_data.as_named_input('test_data')
    ],
    outputs=[
        model_score.as_mount()
    ],
    compute_target=compute,
    runconfig=runconfig
)
```

Then we can just load is as with any other input:

```python
# in evaluate.py
import pickle
from azureml.core.run import Run

run = Run.get_context()

model_file_path = run.input_datasets['model']
with open(os.path.join(model_file_path, 'model.pkl'), 'rb') as file:
    model_in_training = pickle.load(file)
```

## Writing Unit Tests

### Tests Location

You should write your tests inside of the `testing` folder.
Initialisation files for pytest as well as resource files for the coverage or the test requirements file for example should be on the top level of the testing folder.
Any change to the path to that testing folder should be reflected in the pipeline's yaml file.

It is advised to use [tox](https://christophergs.com/python/2020/04/12/python-tox-why-use-it-and-tutorial/) for testing and a template version of what that would look like in this repository will be added in a future version.

### Using Pytest

We are currently using the pytest library for testing. ('https://docs.pytest.org/en/latest/contents.html')
It is important to cover the main elements we are using from pytest in order to better understand the tests' structure in this repository.

#### Using Pytest Fixtures

Pytest fixtures ('https://docs.pytest.org/en/latest/how-to/fixtures.html#how-to-fixtures') are used when you need to use the same variable in different tests.
Here is a sample case:

```python
@pytest.fixture()
def train_test_mock_model(mocker):
    mocker.patch.object(train.Model, 'train_arimax_model',
                        return_value=({}, ts.bidh_ensemble_model_arima_mse_0))
    mocker.patch.object(train.Model, 'train_xgb_model',
                        return_value=({}, ts.bidh_ensemble_model_xgb_mse_0))
    mocker.patch.object(train.Model, 'predict',
                        return_value=ts.bidh_ensemble_model_prediction_0)
    model = train.Model(ts.bidh_ensemble_model_hyperparams_sample_0)
    return model

def test_train(train_test_mock_model: train.Model,
               stats: bool,
               expected: Union[Dict[str, List[Union[int, float]]], str],
               reason: str):
    assert train_test_mock_model.train(
        {'gauge_level': [1, 1, 1, 1]}, stats=stats) == expected, reason
    if stats:
        assert train_test_mock_model.stats == expected, 'self.stats is not equal to expected'

def test_calc_coeffs(train_test_mock_model: train.Model,
                     expected: Dict[str, List[Union[int, float]]],
                     reason: str):
    assert train_test_mock_model.calc_coeffs() == expected, reason

def test_get_features(train_test_mock_model: train.Model,
                      data: pd.DataFrame,
                      expected: List[str],
                      reason: str):
    assert train_test_mock_model.get_features(data) == expected, reason

```

The mocked model will then automatically be initialised and passed to all the tests that come after it and have it as a parameter. You shouldn't use brackets when you call it inside of the tests, the pytest fixtures decorator takes care of all of that for you.

#### Using Pytest Mocker

Mocking ('https://medium.com/analytics-vidhya/mocking-in-python-with-pytest-mock-part-i-6203c8ad3606') is a very important concept in unit testing because it allows you to save a lot of time and permits you to focus on testing your components without relying on the correctness of third party ones.
If we take the same example as before:

```python
@pytest.fixture()
def train_test_mock_model(mocker):
    mocker.patch.object(train.Model, 'train_arimax_model',
                        return_value=({}, ts.bidh_ensemble_model_arima_mse_0))
    mocker.patch.object(train.Model, 'train_xgb_model',
                        return_value=({}, ts.bidh_ensemble_model_xgb_mse_0))
    mocker.patch.object(train.Model, 'predict',
                        return_value=ts.bidh_ensemble_model_prediction_0)
    model = train.Model(ts.bidh_ensemble_model_hyperparams_sample_0)
    return model
```

We can see that we have mocker that is passed as a parameter to that fixture's function. Whenever you use that exact term 'mocker' as a parameter in a test file where you import pytest, pytest will know to use it. Alternatively you can also import it from 'pytest.mock' to use it more explicitly, there is also the 'unittest' library in Python which also has a 'unittest.mock' that could be used.
After getting access to the mocker, you can now start to patch functions, objects or variables. In the example shown, we are using the pytest mocker to patch training and prediction functions of our Model class which is located in our train library that we imported and passing to them the return values we want. This is done in our case because training models takes a relatively long time and unit tests should be quick. If training both the arimax and xgb models in this case takes about 2.5 mins for both given a particular set of input, and we had 4 tests on the '.train()' or '.fit()' method, that's already 10 minutes simply for 4 tests.
It is important to know that if you are mocking a whole object and not just a function of that object, the type of that object becomes MagicMock.objectType, this means that this could cause some issues when using instanceOf() or other functions which rely on the object to be of a certain type so pay attention when doing that.

#### Using Pytest Skipif

Pytest's Skipif ('https://qavalidation.com/2021/01/pytest-options-how-to-skip-or-run-specific-tests.html/') is a very handy functionality for writing generic tests or when writing tests that are based on prior tests (check the following 'https://stackoverflow.com/questions/17571438/test-case-execution-order-in-pytest').

```python
@pytest.mark.skipif(not th.check_func_in_obj('train', train.Model({})),
                    reason='No function called train')
def test_train(train_test_mock_model: train.Model,
               stats: bool,
               expected: Union[Dict[str, List[Union[int, float]]], str],
               reason: str):
```

This example means that pytest should skip the 'test_train' test if there is no function called train in the train.Model object. That check_func_in_obj could also be used on the file level. You can also write more complex statements to check if multiple functions are in the object, if certain expected parameters are in it etc.
If the boolean inside of the skipif returns false, then pytest will just skip the test and alert the developer that a test has been skipped and show the reason why that test was skipped.

#### Using Pytest Parametrize

Pytest Parameterize ('https://docs.pytest.org/en/6.2.x/parametrize.html') is a very important tool which saves a lot of time and avoids redundant code. It is used when you want to repeat the same test multiple times with different parameters.

```python
@pytest.mark.parametrize("expected,reason,status_code,model_preds",
                         [(Exception(),
                           'get_existing_model_predictions response status code error raising' +
                           ' not working properly',
                           404,
                           None),
                          (Exception('Predictions not returned as array.'),
                           'get_existing_model_predictions predictions not returned as array' +
                           ' not working properly',
                           200,
                           {}),
                          ([1, 2, 3],
                           'get_existing_model_predictions predictions returned as array' +
                           ' but with same length as data not working properly',
                           200,
                           [1, 2, 3])
                          ])
def test_get_existing_model_predictions(
        mocker,
        expected: List[Union[int, float]],
        reason: str,
        status_code: int,
        model_preds: Union[List[Union[int, float]], None, Dict]
):
```

As you can see in this example, the arguments of pytest's parametrize function are first a string with comma separated values that exactly match the name of the parameters in the function it is decorating, and a list of tuples of the same size as those parameters and in the same orders as the parameters in the string, so in this case, if we consider a tuple inside that list, the value in index 0 is the value of 'expected', index 1 is 'reason' etc.
Since the list of tuples is made of 3 tuples in this case, this will generate 3 test cases with the provided parameters and using the assertions inside of 'test_get_existing_model_predictions'.

#### Pytest.ini File

The pytest.ini File ('https://docs.pytest.org/en/6.2.x/customize.html') is very handy to have, it lets you give default options to pytest and lets you require a minimum pytest version to run the tests, omit certain files / test cases, give specified test paths etc.
It will only be in effect if you run pytest at the location where that file is.

### Writing a Unit Test

Now that you have an idea of how we are using pytest, you can start writing some unit tests on your own. It is advised for starters to look for tests in the code that look like the one you want to create and copy/paste them then start editing them.
It is advised to have one 'test_' file per file that you want to test and one 'testing_samples_' when applicable.
You should acquaint yourself with how unit tests happen ('https://docs.pytest.org/en/latest/explanation/anatomy.html#test-anatomy') if you need a better understanding of them and always ask yourself if what you're testing is supposed to be tested by you or by the people that made the library you are using. It is advised to test your function when testing combinations of logics from third party, for example:

```python
import numpy as np

def special_calculation(x: float, y: float) -> float:
    return np.sum(np.multiply(x, y), np.divide(x,y))
```

Even though you are using numpy's functions, you should still make sure that the result is what you expect it to be because you are introducing your own logic. You should test the combination of functions and not the individual sum, multiply and divide functions, those will be tested by the people in charge of numpy.

## Using the MLOps Folder Structure

### Root Structure

This is the root folder structure, more details about each complex section can be found in the subsequent categories.

```structure
|   +-- model_a
|   +-- model_b
|   +-- shared_code
|   +-- testing
|   +-- .gitignore
|   +-- LICENSE-CODE (optional)
|   +-- README.md
```

- **.gitignore** - file that is needed for git so that it can ignore certain files, filetypes, folders etc. you can seek guidance on this file [here](https://git-scm.com/docs/gitignore). No sensitive data should ever be pushed to git, if it has to be shared, please try to find other mediums and only share scripts to handle the data in question.
- **LICENSE-CODE** - file that contains a license code for your project. Non-essential, only relevant if you are sharing your repository. You can find guidance on this file [here](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository). If your repository is a private one that only selected people can utilise, please remove this file.
- **README.md** - one of the most important files of a repository is the entry point `README` file. It should contain information about what your repository contains and how to reach that information i.e. _We store in this repository different kinds of flow related predictive models, one of them is model_a (link to model_a README)_. It should be clear enough for someone that does not possess the detailed information about your repository to have an idea of its contents and purpose. Further documentation inside each different model is advised.

### Models

You should have one folder per different model that you want to deploy. The structure should be the same across different models.

```structure
 +-- model_a
|   +-- config
|   +-- data
|   +-- documentation
|   +-- eda
|   +-- notebooks
|   +-- script_steps
|   +-- aml_pipeline.py
|   +-- devops_pipeline.yml
+-- model_b
|   +-- config
     ...
```

- **config** - Contains your entry script as well as different deployment configurations, environments needed and your inference config.
- **data** - This folder should always be git ignored. It contains all kinds of data that is generated by the code. You should **never** share data through Github. All raw data should be stored in a secure commonly accessible location and interim data (i.e. cleaned raw data, training and testing data sets) should be created through the code and kept in this folder locally.
- **documentation** - This folder should contain any relevant documentation on the model. It is advised to have a main documentation file that is linked to the root folder's `README`.
- **eda** - This folder should contain runner scripts or jupyter notebooks that use the functions and classes in the `shared_code` to run your exploratory data analysis processes.
- **notebooks** - This folder should contain jupyter notebooks that do not have a place in the `eda` folder.
- **script_steps, aml_pipeline.py and devops_pipeline.yml** - This folder, files and their contents are explained in detail in the [So you have a new model](#so-you-have-a-new-model) and [Using the AML Pipeline](#using-the-aml-pipeline) sections of this document. What you should note is that these should be unique per model and you should not mix pipelines together unless you are certain that it behaves the way you want it to behave.

### Shared Code

This folder should contain all code that is re-usable across different models in this repository.

```structure
+-- shared_code
|   +-- model_type_function
|   +-- toolbox_a
|   +-- ... 
```

It is best to separate the re-usable code in the following fashion:

- **'model_type'_'function'** - for example `level_model_utils`, `level_model_etl` etc. for utilities related to canal level models. These should be functions and classes that are generalisable and re-usable with canal level models and are specific to them and not any machine learning process or model in general. This code should be tested and should only live in this repository unless there are multiple repositories that use or will use it.
- **'toolbox'** - for example `plottingtoolbox`, `geospatialtoolbox` etc. for code that is not only re-usable on the `model_type` level but on a very general basis. These folders will contain functions that can become candidates to either create a toolbox or be included in an already created one. They do not need to be tested if they are to be included in a toolbox in a short timespan but if they are not they should be tested. If their inclusion to or as a toolbox was rejected, their folder should become `'model_type'_'toolbox'` and they should be tested in this repository.

Please note that if you are trying to access `shared_code` files from a `PythonScriptStep` defined in the `aml_pipeline` file, it is advised to either change the source directory to the root of the project.

### Testing

This folder should contain all your tests for this repository. It should look like this:

```structure
+-- model
|   +-- eda
│       +-- file_1.py
│       +-- file_2.py
+-- shared_code
|   +-- sampletoolbox
|       +-- file_1.py
+-- testing
|   +-- model
|       +-- test_file_1.py
|       +-- test_file_2.py
|       +-- testing_samples
|           +-- samples_file_1.py
|           +-- samples_file_2.py
|           +-- data
|               +-- data_test_case_1.xlsx
|   +-- shared_code
|       +-- sampletoolbox
|           +-- test_file_1.py
|   +-- testing_helpers.py
|   +-- pytest.ini
|   +-- .coveragerc
|   +-- .pylintrc
```

You should always test as much of your code as you can but in certain cases, there is no need to do so. For example, for runner files that only contain code under `if __name__ == 'main'` if everything in it that could be refactored as a function already is and is tested, there is no need to have a test file for it.

### VirtualEnvs

When possible, it is best to work with virtual environments to reduce issues arising from the usage of the wrong packages, python version etc. In order to facilitate this, it is advised to create a `virtualenvs` folder which will contain all your virtual environments. You can refer to [this document](https://techinscribed.com/python-virtual-environment-in-vscode/#:~:text=Add%20the%20Virtual%20Environment%20Folder%20to%20VSCode.%20Add,the%20interpreter%20version%20displayed%20on%20the%20left-bottom%20corner.) to learn how to setup your virtual environments with VSCode.

When using multiple virtual environments, it is better to give them trackable names i.e. `.venv_modelA` instead of `.venv`.

Virtual environments should **never** be pushed to git, they should always be git ignored.
