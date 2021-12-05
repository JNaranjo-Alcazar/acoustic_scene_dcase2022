FROM python:3.7

WORKDIR /audio_asc

COPY requirements.txt .

RUN python3.7 -m pip install -r requirements.txt
