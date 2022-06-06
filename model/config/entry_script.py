"""
    This script is used to load the model from its pickled state
    and to define the basic data i/o for it.

    We have two functions, and a global variable.

    The global variable `MODEL` is used to store
    the working instance of the trained model.

    The first function is the `init()` with the purpose of
    loading the pickled model, and storing it
    into the global variable `MODEL`.

    The second function `run(data)` takes input from a HTTP request
    (all very neatly abstracted away for us) and feeds it into the
    model stored as `MODEL`. This requires processing `data` as if you
    just received it as a request body, and returning the result.

    The workflow we are defining here looks something like this:
      1. container deploys -> `init()`
         prepares model -> ready to receive request
      2. request received -> `run(data)`
         gets model prediction/result -> prediction/result returned

"""

from pathlib import Path
import json
import numpy as np
from tensorflow import keras

# global variable to hold model instance after it has been unpickled
MODEL = None


# should-change -- based on the type of your model you might want
# to use dill/ joblib instead
def init():
    """
    Initialise by loading pickled model into global variable `MODEL`.
    """
    global MODEL
    MODEL = keras.models.load_model(Path("AZUREML_MODEL_DIR") / Path("model/model.tf"))


# could-change -- based on how you want to score your model,
# this could change however this should cover
# a good range of model outputs.
def run(data):
    """On receiving data, return the model prediction.
    The return value of this function will be
    sent to the client that made the request for prediction.
    """
    try:

        # these two steps may look different for different projects,
        # as they are model specific
        data = np.array(json.loads(data))
        result = MODEL.predict(data)

        # you can return any data type, as long as it is JSON serializable
        return result.tolist()

    # note that this generic error handling is great in development,
    # but be aware that _any_ returned value will be sent to the client.
    except Exception as err:
        return str(err)
