import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType


data_sample = PandasParameterType(pd.DataFrame({"gender": pd.Series(["example_value"], dtype="object"), "SeniorCitizen": pd.Series([False], dtype="bool"), "Partner": pd.Series([False], dtype="bool"), "Dependents": pd.Series([False], dtype="bool"), "tenure": pd.Series([0], dtype="int64"), "PhoneService": pd.Series([False], dtype="bool"), "MultipleLines": pd.Series(["example_value"], dtype="object"), "InternetService": pd.Series(["example_value"], dtype="object"), "OnlineSecurity": pd.Series(["example_value"], dtype="object"), "OnlineBackup": pd.Series(["example_value"], dtype="object"), "DeviceProtection": pd.Series(["example_value"], dtype="object"), "TechSupport": pd.Series(["example_value"], dtype="object"), "StreamingTV": pd.Series(["example_value"], dtype="object"), "StreamingMovies": pd.Series(["example_value"], dtype="object"), "Contract": pd.Series(["example_value"], dtype="object"), "PaperlessBilling": pd.Series([False], dtype="bool"), "PaymentMethod": pd.Series(["example_value"], dtype="object"), "MonthlyCharges": pd.Series([0.0], dtype="float64"), "TotalCharges": pd.Series([0.0], dtype="float64")}))
input_sample = StandardPythonParameterType({'data': data_sample})
method_sample = StandardPythonParameterType("predict")
sample_global_params = StandardPythonParameterType({"method": method_sample})

result_sample = NumpyParameterType(np.array([False]))
output_sample = StandardPythonParameterType({'Results':result_sample})


def get_model_root(model_root: str):
    root_contents = os.listdir(model_root)
    logger.info(f"List model root dir: {os.listdir(model_root)}")
    if len(root_contents) == 1:
        root_file_path = os.path.join(model_root, root_contents[0])
        return root_file_path if os.path.isdir(root_file_path) else model_root
    else:
        raise Exception("Unexpected. root must contain a model file or a mlflow model directory")


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_root = get_model_root(os.getenv('AZUREML_MODEL_DIR'))
    model_path = os.path.join(model_root, 'best_hyperdrive_model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('GlobalParameters', sample_global_params, convert_to_provided_type=False)
@input_schema('Inputs', input_sample)
@output_schema(output_sample)
def run(Inputs, GlobalParameters={"method": "predict"}):
    data = Inputs['data']
    if GlobalParameters.get("method", None) == "predict_proba":
        result = model.predict_proba(data)
    elif GlobalParameters.get("method", None) == "predict":
        result = model.predict(data)
    else:
        raise Exception(f"Invalid predict method argument received. GlobalParameters: {GlobalParameters}")
    if isinstance(result, pd.DataFrame):
        result = result.values
    return {'Results':result.tolist()}
