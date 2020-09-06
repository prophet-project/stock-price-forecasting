FROM tensorflow/tensorflow:2.2.0-gpu as base

RUN apt update && \
    apt install -y make git

RUN pip install spacy

# Install spacy models
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_md && \
    python -m spacy download en_core_web_lg -v

FROM base as second

RUN pip install \
    tensorflow_datasets \
    dvc pydrive2

FROM second

# Data visualisation
RUN pip install \
    plotly==4.9.0 cufflinks chart_studio \
    pandas \
    "notebook>=5.3" \
    "ipywidgets>=7.2"

WORKDIR /work
