FROM tensorflow/tensorflow:2.4.0-gpu as base

RUN apt update && \
    apt install -y make git

FROM base as second

COPY ta-lib-0.4.0-src.tar.gz ta-lib-0.4.0-src.tar.gz

RUN tar -xzf ta-lib-0.4.0-src.tar.gz
RUN cd ta-lib/ && ./configure --prefix=/usr && make && make install

FROM second as third

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

FROM third

WORKDIR /work

CMD [ "make", "notebook" ]