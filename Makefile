#!/usr/bin/env make

.PHONY: train docker-build docker-console spacy-load-model

DOCKER_IMAGE_VERSION=0.6.0
DOCKER_IMAGE_TAG=leovs09/sentiment:$(DOCKER_IMAGE_VERSION)

# ---------------------------------------------------------------------------------------------------------------------
# DEVELOPMENT
# ---------------------------------------------------------------------------------------------------------------------

# Will reproduce all stages to generate model based on changes
default:
	dvc repro

# Will load data and models which specifaed by dvc files
checkout:
	dvc pull

train:
	echo "-m flag will run script in module mode, which accept relative imports"
	python -m src.train

test:
	echo "-m flag will run script in module mode, which accept relative imports"
	python -m src.test

# Will load medium size model for spacy
spacy-load-md:
	python -m spacy download en_core_web_md

# Will start notebook environment on http://0.0.0.0:8888
notebook: 
	jupyter notebook --ip=0.0.0.0 --allow-root

# ---------------------------------------------------------------------------------------------------------------------
# DOCKER
# ---------------------------------------------------------------------------------------------------------------------

# Will build docker image for development
docker-build:
	docker build --tag $(DOCKER_IMAGE_TAG) .

# Will start in docker develoment environment
docker-console:
	docker run -it --rm -v ${PWD}:/work -w /work -p 8888:8888 $(DOCKER_IMAGE_TAG) bash


