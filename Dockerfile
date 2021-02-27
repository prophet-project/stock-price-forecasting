FROM tensorflow/tensorflow:2.4.0-gpu as base

RUN apt update && \
    apt install -y make git

FROM base as second

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

FROM second

WORKDIR /work

CMD [ "make", "notebook" ]