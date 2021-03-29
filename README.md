# Stock price forecasting

Research of NN model for market stock price forecasting

Target dataset [Bitcoin in Cryptocurrency Historical Prices](https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory?select=coin_Bitcoin.csv)

Bitcoin data at 1-day intervals from April 28, 2013

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

You can browse research as [notebook](https://github.com/prophet-project/stock-price-forecasting/blob/master/index.ipynb)

if github not load notebook you can open [html version](https://prophet-project.github.io/stock-price-forecasting/)
or even [python version](https://github.com/prophet-project/stock-price-forecasting/blob/master/results/index.py)

## Development

If you on Windows, build and run in-Docker development enviroment

```bash
# Build image
make docker-build
# Start docker container, map volume on current folder and attach current console
make docker-console
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

## DVC

[DVC](https://dvc.org/) is  version control for datasets and neural networks

### First Setup

For start control files and setup storage

```bash
# Add `data` folder or file to version control
dvc add data
# Add remove storage in Google Drive
# Folder id can be found in url of Google Drive when open folder
dvc remote add --default myremote gdrive://{folder_id}/some_folder_inside
# Push data to storage
dvc push
```

### Checkout

After setup you can checkout model and dataset

```bash
dvc checkout 
```
