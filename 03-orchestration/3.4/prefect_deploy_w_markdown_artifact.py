import pathlib
import pickle
import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow
import xgboost as xgb
from prefect import flow, task,artifacts

from prefect.artifacts import create_markdown_artifact


@task(retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pd.DataFrame:
    """
    This task reads data from a parquet file and performs preprocessing operations on the data.
    It converts datetime columns to datetime format, calculates the duration between pickup and dropoff times,
    filters the data based on duration constraints, and converts categorical columns to string format.

    Args:
        filename (str): The path to the parquet file.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Read the data from the parquet file
    df = pd.read_parquet(filename)

    # Convert datetime columns to datetime format
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    # Calculate the duration between pickup and dropoff times
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    # Filter the data based on duration constraints
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Convert categorical columns to string format
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df



@task
def add_features(
    df_train: pd.DataFrame, df_val: pd.DataFrame
) -> tuple(
    [
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        np.ndarray,
        np.ndarray,
        sklearn.feature_extraction.DictVectorizer,
    ]
):
    """
    This task performs feature engineering and transforming the categorical
    and numerical features into a suitable format for model training.

    Args:
        df_train (pd.DataFrame): The training dataset.
        df_val (pd.DataFrame): The validation dataset.

    Returns:
        tuple: A tuple containing the transformed features and target variables, along with the
        DictVectorizer object used for feature transformation.
    """
    
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv


@task(log_prints=True)
def train_best_model(
    X_train: scipy.sparse._csr.csr_matrix,
    X_val: scipy.sparse._csr.csr_matrix,
    y_train: np.ndarray,
    y_val: np.ndarray,
    dv: sklearn.feature_extraction.DictVectorizer,
) -> None:
    """
    This task trains an XGBoost model with the specified hyperparameters and logs the model
    parameters, evaluation metrics, and artifacts using MLflow.

    Args:
        X_train (scipy.sparse._csr.csr_matrix): The sparse matrix of transformed features for training.
        X_val (scipy.sparse._csr.csr_matrix): The sparse matrix of transformed features for validation.
        y_train (np.ndarray): The target variables for training.
        y_val (np.ndarray): The target variables for validation.
        dv (sklearn.feature_extraction.DictVectorizer): The DictVectorizer object used for feature transformation.

    Returns:
        None
    """
    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "objective": "reg:linear",
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42,
        }

        # Log the best hyperparameters
        mlflow.log_params(best_params)

        # Train the model with early stopping
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=20,
        )

        # Make predictions on the validation data
        y_pred = booster.predict(valid)

        # Calculate the RMSE
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        # Log the RMSE as a metric
        mlflow.log_metric("rmse", rmse)

        # Create a directory to store the artifacts
        pathlib.Path("models").mkdir(exist_ok=True)

        # Save the DictVectorizer object
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        # Log the DictVectorizer artifact
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        # Log the XGBoost model as an artifact
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

    return rmse


@task
def create_rmse_artifact(rmse: float) -> None:
    """
    This task generates a Markdown artifact that showcases the validation RMSE value.
    It takes the RMSE value as input and creates a Markdown artifact with the formatted content.

    Args:
        rmse (float): The validation RMSE value.

    Returns:
        None
    """
    # Create the content for the Markdown artifact
    markdown_content = f"RMSE for the validation data: **{rmse}**"

    # Create the Markdown artifact
    create_markdown_artifact(
        key="validation-rmse",
        markdown=markdown_content,
        description="RMSE for Validation Data",
    )

    return None


@flow
def main_flow_w_artifact(
    train_path: str = "./data/green_tripdata_2023-02.parquet",
    val_path: str = "./data/green_tripdata_2023-03.parquet",
) -> None:
    """
    The main training pipeline.

    This flow represents the main training pipeline.
    It sets up the MLflow configurations, loads the data, performs transformation, trains the model,
    and generates the RMSE artifact.

    Args:
        train_path (str): The path to the training data parquet file.
        val_path (str): The path to the validation data parquet file.

    Returns:
        None
    """
    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")

    # Load the data
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # Perform data transformation
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Train the model and get the RMSE
    rmse = train_best_model(X_train, X_val, y_train, y_val, dv)
    
    # Generate the RMSE artifact
    create_rmse_artifact(rmse)


from prefect import flow
from prefect.deployments import Deployment

def deploy():
    deployment = Deployment.build_from_flow(
        flow=main_flow_w_artifact,
        name="deploy with markdown artifact"
    )
    deployment.apply()

if __name__ == "__main__":
    deploy()