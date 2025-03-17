FROM python:3.10-slim

ARG API_KEY
ENV API_KEY=$API_KEY

COPY app.py ./app.py
COPY requirements/requirements-app.txt ./requirements-app.txt
COPY output/model.joblib ./output/model.joblib
COPY src/model/predictor.py ./src/model/predictor.py
COPY src/model/preprocessing.py ./src/model/preprocessing.py

RUN pip install --no-cache-dir -r requirements-app.txt

ENV PORT=8000
EXPOSE $PORT

CMD gunicorn --workers=1 --bind 0.0.0.0:$PORT app:app
