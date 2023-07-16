from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import os,sys
import boto3,s3fs
from io import BytesIO
from batch_reconfigured import main

load_dotenv()

categorical = ['PULocationID', 'DOLocationID']

YEAR=2022
MONTH=1
def prepare_data(df):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    def get_input_path(year, month):
        default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
        input_pattern = os.getenv('INPUT_FILE_PATTERN')
        print(input_pattern)
        temp = input_pattern.format(year=year, month=month)
        print(temp)
        return temp
    print("before write **** ")
    input_file = get_input_path(YEAR, MONTH)

    AWS_REGION = 'us-east-1'
    AWS_BUCKET_NAME = os.environ.get('AWS_BUCKET_NAME')
    ENDPOINT_URL = os.environ.get('S3_ENDPOINT')
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY_ID')

    boto3.setup_default_session()
    s3_client = boto3.client("s3", region_name=AWS_REGION, endpoint_url=ENDPOINT_URL,aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    ## Write to localfirst and then write to s3
    #https://hands-on.cloud/testing-python-aws-applications-using-localstack/#h-upload-file-to-s3-bucket
    def upload_file1(df,file_name, bucket, object_name=None):
        """
        Upload a file to a S3 bucket.
        """
        try:
            if object_name is None:
                object_name = os.path.basename(file_name).replace('s3://nyc-duration/', '')
                df.to_parquet(object_name, engine='pyarrow', index=False)

            with open(object_name, 'rb') as f:
                object_data = f.read()
                s3_client.put_object(Body=object_data, Bucket=bucket, Key='upload_file1.parquet')
        except Exception:
            raise

    ## Write to s3 directly
    def upload_file(df, file_name, bucket,  object_name=None):
        """
        Upload a DataFrame as a Parquet file to an S3 bucket.
        """
        try:
            # Write the DataFrame to a Parquet file in memory
            buffer = BytesIO()
            df.to_parquet(buffer, engine='pyarrow', index=False)
            buffer.seek(0)  # Reset the buffer position to the beginning

            # Upload the Parquet file to S3
            if object_name is None:
                object_name = os.path.basename(file_name).replace('s3://nyc-duration/', '')

            s3_client.put_object(Body=buffer, Bucket=bucket, Key=object_name)
        except Exception as e:
            raise e


    # Checking if the bucket already exists
    for bucket in s3_client.list_buckets()['Buckets']:
        print("Existing bucket",bucket['Name'])

        if AWS_BUCKET_NAME not in [bucket['Name'] for bucket in s3_client.list_buckets()['Buckets']]:
            print("Creating the bucket...")
            s3_client.create_bucket(Bucket=AWS_BUCKET_NAME)

    upload_file(df, input_file, AWS_BUCKET_NAME)
    upload_file1(df, input_file, AWS_BUCKET_NAME)

    #https://stackoverflow.com/questions/56566785/how-to-use-boto3-to-retrieve-s3-file-size
    def get_file_size(file_name):
        object_name = os.path.basename(file_name)
        response = s3_client.head_object(
            Bucket=AWS_BUCKET_NAME,
            Key=object_name
        )
        print(" size of the file ::",response['ContentLength'])

    get_file_size(input_file)

    #predict and save the results to output file
    def save_data(year, month):
        df_result,output_file = main(YEAR,MONTH)
        print("output_file ::",output_file)
        upload_file(df_result, output_file, AWS_BUCKET_NAME)

    save_data(YEAR, MONTH)
