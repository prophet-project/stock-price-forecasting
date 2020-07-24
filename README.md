# Sentiment Analyse

Sentiment Analyse Neural Network Model

Can classificate given text as bad or good review.

Based on Spacy - for text embedding. Spacy contain big multilanguage dataset, and can build 300 dimensions vector on text entityes.
Model build on Tensorflow - easy for build network model fraemwork, but possible for production requires additional tuning.

## Requirements

Can be run in docker enviroment, or if you on linux:

* Python - version 3.*
* Spacy - version 2.*
* Tensorflow - version 2.2 or bigger
* Make - optional, all commands can be run manually
* DVC - Git for machine learning

All Python libraries and models described in Dockerfile

## Development

If you on Windows, build and run in-Docker development enviroment

```bash
# Build image
make docker-build
# Start docker container, map volume on current folder and attach current console
make docker-console
```

If you not in docker you need load Spacy model

```bash
# Will load meduim size model, good for developement, but for production better load lardger
make spacy-load-md
```

For rebuild model or test it after data or code changes, just run

```bash
make
# or
dvc repro
```

it will calculate all affected files and changed stages and start rebuild process

## Training

For train model just run

```bash
make train
```

It will load dataset and start training

## Testing

For test model just run

```bash
make test
```

## Jupiter Notebook

In docker image insatlled Jupiter Notebook

For run notebook

```bash
make notebook
```

## Five main steps for build Neural Network

1) Load dataset - split into train and test, and validation - which need for check training progress
2) Normalise data - need normalise text to vector with constant size, for put it into first layer of network
3) Build network model - stack layers for build model of network
4) Train - put train data with correct hyperparameters
5) Test - for insure correctness
