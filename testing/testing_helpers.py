"""
Contributors: Antoine Chammas

Summary: This file contains helper functions that can be used
by multiple test files.
"""
import os
from azureml.core.workspace import Workspace

###########
# Helpers #
###########


def check_func_in_obj(func_name: str, model: object) -> bool:
    """
    Checks if a function is implemented in an object, if so
    return true, else returns false.

    Args:
        func_name (str): [description]
        model (object): [description]

    Returns:
        bool: [description]
    """
    att = getattr(model, func_name, None)
    if callable(att):
        return True
    return False


def check_atr_vals_in_obj(all_atrs: dict, model: object) -> list:
    """
    Tests that attribute values are present in an object
    Args:
        all_attrs (list): [description]
        model (object): [description]

    Returns:
        list: [description]
    """
    errors = []
    for atr, value in all_atrs.items():
        if not hasattr(model, atr):
            errors.append("Attribute not present - " + str(atr))
            continue
        if getattr(model, atr) != value:
            errors.append("Error in value matching, attribute: " + str(atr))
    return errors


def check_atr_in_obj(all_atrs: list, model: object) -> list:
    """
    Tests that a list of attributes is present in
    Args:
        all_attrs (list): [description]
        model (object): [description]

    Returns:
        list: [description]
    """
    errors = []
    for atr in all_atrs:
        if not hasattr(model, atr):
            errors.append("Attribute not present - " + str(atr))
    return errors


def get_env_var(name, errors=None, default=None):
    """
    Function used to get an environment variable.

    Parameters:
        name (String): Case sensitive name of the environment variable
                       you want to get.
        errors (List): The errors list in the tests to be populated with
                       an exception if one occurs.
                       Defaults to [].
    Returns:
        If an environment variable corresponding to the given name exists:
        The environment variable corresponding to the given name.
        If it doesn't:
        None
    """
    if errors is None:
        errors = []
    try:
        return os.environ[name]
    except Exception as exception:
        errors.append(
            "Couldn't load Environment Variable " + name + ", Error: " + str(exception)
        )
    return default


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
