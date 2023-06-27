FROM svizor/zoomcamp-model:mlops-3.10.0-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app 

COPY Pipfile Pipfile.lock requirements.txt ./

RUN pipenv install --system --deploy
RUN pip install --no-cache-dir -r requirements.txt

COPY starter.py ./