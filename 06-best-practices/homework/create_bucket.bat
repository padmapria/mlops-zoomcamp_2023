@echo off

set S3_ENDPOINT=http://localhost:4566
set BUCKET_NAME=nyc-duration
set AWS_REGION=us-east-1
set AWS_CLI_VERSION=2

@echo Checking if the bucket already exists...
aws --region %AWS_REGION% s3 mb s3://nyc-duration --endpoint-url %S3_ENDPOINT% --no-sign-request > nul 2>&1

if errorlevel 1 (
    @echo Creating the bucket...
    aws --region %AWS_REGION% s3 mb s3://nyc-duration --endpoint-url %S3_ENDPOINT% --no-sign-request
) else (
    @echo The bucket already exists.
)

@echo Listing the contents of the bucket...
aws --region %AWS_REGION% s3 ls s3://%BUCKET_NAME% --endpoint-url %S3_ENDPOINT% --no-sign-request

pause

