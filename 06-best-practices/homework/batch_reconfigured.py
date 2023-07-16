#!/usr/bin/env python
# coding: utf-8

import sys,os,boto3
import pickle
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
input_file_pattern = os.getenv("INPUT_FILE_PATTERN")
output_file_pattern = os.getenv("OUTPUT_FILE_PATTERN")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
ENDPOINT_URL = os.environ.get('S3_ENDPOINT')

def read_data(input_file,categorical):
    options = {
    'client_kwargs': {
        'endpoint_url': ENDPOINT_URL
        }
    }

    df = pd.read_parquet(os.path.basename(input_file).replace('s3://nyc-duration/', ''))
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN')
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 'taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN')
    return output_pattern.format(year=year, month=month)


def main(year, month):
    input_file = get_input_path(year, month)
    print("input_file",input_file)
    output_file = get_output_path(year, month)
    
    categorical = ['PULocationID', 'DOLocationID']
    
    df = read_data(input_file,categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    
    # Construct the full path to the model file
    model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.bin')
    
    with open(model_file, 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    print('predicted duration sum:', df_result['predicted_duration'].sum())
    
    return df_result,output_file