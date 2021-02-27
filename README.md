# Stock price forecasting

Research of NN model for market stock price forecasting

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

You can browse research as [notebook](https://github.com/LeoVS09/stock-price-forecasting/blob/master/analyse.ipynb)

if github not load notebook you can open [html version](https://leovs09.github.io/stock-price-forecasting)
or even [python version](https://github.com/LeoVS09/stock-price-forecasting/blob/master/results/anayse.py)

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
