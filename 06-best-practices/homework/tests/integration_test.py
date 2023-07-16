from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import os,sys
import boto3
from io import BytesIO
from batch_reconfigured import main

load_dotenv()
YEAR=2022
MONTH=1

categorical = ['PULocationID', 'DOLocationID']

def prepare_data(df):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)

## Liveserver s3 configuration
AWS_REGION = 'us-east-1'
ENDPOINT_URL = os.environ.get('S3_ENDPOINT')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_BUCKET_NAME = os.environ.get('AWS_BUCKET_NAME')

boto3.setup_default_session()
s3_client = boto3.client("s3", region_name=AWS_REGION, endpoint_url=ENDPOINT_URL,aws_access_key_id=AWS_ACCESS_KEY_ID,
aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# Checking if the bucket already exists
def create_bucket():
    for bucket in s3_client.list_buckets()['Buckets']:
        print("Existing bucket",bucket['Name'])

        if AWS_BUCKET_NAME not in [bucket['Name'] for bucket in s3_client.list_buckets()['Buckets']]:
            print("Creating the bucket...")
            s3_client.create_bucket(Bucket=AWS_BUCKET_NAME)

def check_files_in_bucket():
    # List the objects in the bucket
    print("Files in the bucket ... ")
    response = s3_client.list_objects_v2(Bucket=AWS_BUCKET_NAME)

    # Retrieve the keys of the objects
    keys = [obj['Key'] for obj in response.get('Contents', [])]

    # Print the keys
    for key in keys:
        print(key)

    if not keys:
        print("The bucket is empty.")


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
    ## create the input dataframe
    df = prepare_data(df)


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
                object_name = file_name.replace('s3://nyc-duration/', '')
 
            s3_client.put_object(Body=buffer, Bucket=bucket, Key=object_name)
        except Exception as e:
            raise e


    ## Defining the filename to store in s3
    def get_input_path(year, month):
        default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
        input_pattern = os.getenv('INPUT_FILE_PATTERN')
        return input_pattern.format(year=year, month=month)

    input_file = get_input_path(YEAR, MONTH)

    upload_file(df, input_file, AWS_BUCKET_NAME)
    #upload_file1(df, input_file, AWS_BUCKET_NAME)

    ## Q5
    #https://stackoverflow.com/questions/56566785/how-to-use-boto3-to-retrieve-s3-file-size
    def get_file_size(file_name):
        object_name = file_name.replace('s3://nyc-duration/', '')
        print("File name:: ",object_name)
        response = s3_client.head_object(
            Bucket=AWS_BUCKET_NAME,
            Key=object_name
        )
        print(" size of the file ::",response['ContentLength'])

    get_file_size(input_file)

    ## Q6
    #predict and save the results to output file
    def save_data(year, month):
        check_files_in_bucket()
        df_result,output_file = main(YEAR,MONTH)
        print("output_file ::",output_file)
        upload_file(df_result, output_file, AWS_BUCKET_NAME)
        check_files_in_bucket()

    save_data(YEAR, MONTH)
