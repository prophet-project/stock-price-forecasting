FROM tensorflow/tensorflow:2.2.0-gpu as base

RUN apt update && \
    apt install -y make git

# Required for JamSpell
RUN apt install swig3.0 locales -y
RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

# Required for pycontractions > language_check > LanguageTool
RUN apt install default-jdk

RUN pip install spacy

# Install spacy models
RUN python -m spacy download en_core_web_lg -v

FROM base as second

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

FROM second

WORKDIR /work
