FROM tensorflow/tensorflow:latest-py3 as base

RUN apt update && \
    apt install -y make git

RUN pip install spacy

# Install spacy models
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_md && \
    python -m spacy download en_core_web_lg

FROM base as second

RUN pip install \
    tensorflow==2.2 \
    tensorflow_datasets \
    dvc && pydrive2

FROM second

WORKDIR /work
