FROM tensorflow/tensorflow:2.2.0-gpu as base

RUN apt update && \
    apt install -y make git

RUN pip install spacy

# Install spacy models
RUN python -m spacy download en_core_web_lg -v

FROM base as second

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

FROM second

WORKDIR /work
