@echo off

setlocal

set S3_ENDPOINT=http://localhost:4566
set BUCKET_NAME=nyc-duration

aws s3 ls --endpoint-url=%S3_ENDPOINT% --no-sign-request

aws s3 rb s3://%BUCKET_NAME% --endpoint-url=%S3_ENDPOINT% --no-sign-request --force

aws s3 ls --endpoint-url=%S3_ENDPOINT% --no-sign-request
pause
