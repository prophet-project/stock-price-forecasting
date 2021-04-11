#!/usr/bin/env make

.PHONY: train docker-build docker-console notebook install save-dependencies dev

PROJECT_NAME=stock-price-forecasting
DOCKER_IMAGE_VERSION=0.4.0
DOCKER_IMAGE_TAG=leovs09/$(PROJECT_NAME):$(DOCKER_IMAGE_VERSION)

# ---------------------------------------------------------------------------------------------------------------------
# DEVELOPMENT
# ---------------------------------------------------------------------------------------------------------------------

MODIN_ENGINE=dask

# Will reproduce all stages to generate model based on changes
default:
	dvc repro && dvc push

# Will load data and models which specifaed by dvc files
checkout:
	dvc pull

train:
	echo "-m flag will run script in module mode, which accept relative imports"
	python -m src.train

test:
	echo "-m flag will run script in module mode, which accept relative imports"
	python -m src.test

# Will start notebook environment on http://0.0.0.0:8888
notebook: 
	jupyter notebook --ip=0.0.0.0 --allow-root

install:
	pip install -r requirements.txt

# Will save all current dependencies to requirements.txt
save-dependencies:
	pip freeze > requirements.txt

# docs folder required for github pages
notebook-to-html:
	jupyter nbconvert ./*.ipynb --to html --output-dir="./docs"

notebook-to-python:
	jupyter nbconvert ./*.ipynb --to python --output-dir="./results"

notebook-artifacts: notebook-to-html notebook-to-python chmod

metrics-diff:
	dvc metrics diff

# ---------------------------------------------------------------------------------------------------------------------
# DOCKER
# ---------------------------------------------------------------------------------------------------------------------

CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)

dev: docker-build docker attach-console

# Will build docker image for development
docker-build:
	docker build --tag $(DOCKER_IMAGE_TAG) .

# Will start in docker develoment environment
docker-console:
	docker-compose run --service-ports notebook bash

console: docker-console

docker:
	docker-compose run -d --service-ports notebook

docker-it:
	docker-compose run --service-ports notebook 

attach-console:
	docker exec -it notebook bash

chmod:
	chmod -R 777 .

mlflow-ui:
	docker-compose run --service-ports mlflow-ui
