FROM tensorflow/tensorflow:latest-py3

RUN pip install spacy make tensorflow==2.2 tensorflow_datasets

# Install spacy models
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_md && \
    python -m spacy download en_core_web_lg

WORKDIR /work
