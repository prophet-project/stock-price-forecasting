# Sentiment Analyse

Sentiment Analyse Neural Network Model

Can classificate given text as bad or good review.

Target dataset is [Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/kazanova/sentiment140)

## Text preprocessing

Will use multiple pre trained embeding vectors for different tasks.

For tokenization and base lingustic analyse use spacy `en_core_web_lg` model.
Large english corpus. Will be downloaded automatically by space in Docker.

For misspelling and contractions replacement will use `GoogleNews-vectors-negative300.bin`
as biggest open accessable embeding corpus.
It must be enought for all tasks, but if for you task other corpus will be enough use it.

List of most accessable embeding corpuses and tool for download them
there: <https://github.com/RaRe-Technologies/gensim-data>

### GoogleNews-vectors-negative300

Pre-trained vectors trained on a part of the Google News dataset (about 100 billion words).
The model contains 300-dimensional vectors for 3 million words and phrases.

<https://code.google.com/archive/p/word2vec/>

## Requirements

Can be run in docker enviroment, or if you on linux:

* Python - version 3.*
* Tensorflow - version 2.2 or bigger
* Make - optional, all commands can be run manually
* DVC - Git for machine learning, need for load datasets and trained models.

All other Python libraries and models described in `Dockerfile` and `requirements.txt`

### Hardware

* NVidia videocard with DLSS version >= 10 - actually GPU optional,
    and learning can be run on CPU,
    but models and enviroment configurated to run on GPU,
    in base case tenserflow can fallback to CPU,
    so not need change anything for start development

## Research Results

You can browse research as [notebook](https://github.com/LeoVS09/sentiment/blob/master/analyse.ipynb)

if github not load notebook you can open [html version](https://leovs09.github.io/sentiment)
or even [python version](https://github.com/LeoVS09/sentiment/blob/master/results/anayse.py)

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
