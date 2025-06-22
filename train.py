from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace, Dataset

def clean_data(data):
    pm = {"Electronic check": 1, "Mailed check": 2, "Bank transfer (automatic)": 3, "Credit card (automatic)": 4}
    contract = {"Month-to-month": 1, "One year": 2, "Two year": 3}
    service = {"No internet service": 1, "Yes": 2, "No": 3}
    phone = {"No phone service": 1, "Yes": 2, "No": 3}
    internet = {"Fiber optic": 1, "DSL": 2, "No": 3}

    # data.drop(columns=['customerID'], inplace=True)
    x_df = data.dropna()

    x_df["gender"] = x_df.gender.apply(lambda s: 1 if s == "Male" else 0)
    # x_df["Partner"] = x_df.Partner.apply(lambda s: 1 if s == "Yes" else 0)
    # x_df["Dependents"] = x_df.Dependents.apply(lambda s: 1 if s == "Yes" else 0)
    # x_df["PhoneService"] = x_df.PhoneService.apply(lambda s: 1 if s == "Yes" else 0)
    x_df["MultipleLines"] = x_df.MultipleLines.map(phone)
    x_df["InternetService"] = x_df.InternetService.map(internet)
    x_df["OnlineSecurity"] = x_df.OnlineSecurity.map(service)
    x_df["OnlineBackup"] = x_df.OnlineBackup.map(service)
    x_df["DeviceProtection"] = x_df.DeviceProtection.map(service)
    x_df["TechSupport"] = x_df.TechSupport.map(service)
    x_df["StreamingTV"] = x_df.StreamingTV.map(service)
    x_df["StreamingMovies"] = x_df.StreamingMovies.map(service)
    x_df["Contract"] = x_df.Contract.map(contract)
    # x_df["PaperlessBilling"] = x_df.PaperlessBilling.apply(lambda s: 1 if s == "Yes" else 0)
    x_df["PaymentMethod"] = x_df.PaymentMethod.map(pm)
    # x_df["Churn"] = x_df.Churn.apply(lambda s: 1 if s == "Yes" else 0)
    y_df = x_df.pop("Churn")

    print(y_df.value_counts())
    return x_df, y_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0,
                        help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    args = parser.parse_args()
    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    ws = run.experiment.workspace
    
    try:
        ds = Dataset.get_by_name(ws, name='ibm-telco-data', version='1').to_pandas_dataframe()
        print(f"Using existing dataset '{registered_dataset.name}', version {registered_dataset.version}")
    except Exception as get_e:
        print(f"Failed to get existing dataset after registration error: {get_e}")

    x, y = clean_data(ds)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)

    model_filename = 'model.pkl'
    model_path = os.path.join(output_dir, model_filename)

    print(f"Saving model to {model_path}")
    joblib.dump(value=model, filename=model_path)
    print("Model saved successfully in outputs folder.")


if __name__ == '__main__':
    main()