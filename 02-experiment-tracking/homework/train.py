import os
import pickle
import click
import requests

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#library for mlflow
import mlflow
    
#select a SQLite db for the backend store
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
   
    with mlflow.start_run():
        
        #Setting the tags
        mlflow.set_tag("developer", "padmapriya")
        
        #Enabling the autolog with MLFlow
        mlflow.sklearn.autolog()
        
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        
def is_mlflow_server_up(mlflow_server_url):
    try:
        response = requests.get(mlflow_server_url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

if __name__ == '__main__':
    # Example usage
    run_train()
