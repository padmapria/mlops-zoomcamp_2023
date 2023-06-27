#!/usr/bin/env python
# coding: utf-8

import pickle,os
import pandas as pd
import sys

#To pass year and month values via cli
if len(sys.argv) != 3:
    print("Please provide year month.")
else:
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    print("year:", year)
    print("month:", month)
    
#year = 2022
#month = 2
taxi_type = 'yellow'

# Load the model file
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Read the input file
categorical = ['PULocationID', 'DOLocationID']
def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    ## creating ride_id column
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    return df

#Dir path for output file
directory_path = f"output/{taxi_type}"

# Create the directory if it doesn't exist
os.makedirs(directory_path, exist_ok=True)

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
df = read_data(input_file)

output_file = f'{directory_path}/{year:04d}-{month:02d}.parquet'

# Make predictions
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# standard deviation of the predicted duration
import numpy as np
# Calculate mean
mean = np.mean(y_pred)
print("Mean predicted duration :", mean)

# Calculate squared differences
squared_diff = [(pred - mean) ** 2 for pred in y_pred]

# Calculate variance
variance = np.mean(squared_diff)

# Calculate standard deviation
std_deviation = np.sqrt(variance)

print("Standard Deviation:", std_deviation)


# ### Q2. Preparing the output
df_result = df[['ride_id']]
df_result.loc[:, 'predictions'] = y_pred

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

##upload file to s3
import boto3

s3_client = boto3.client('s3')
bucket_name = 'mlops_data_priya'
# Upload the file to S3
s3_client.upload_file(output_file, bucket_name, output_file)