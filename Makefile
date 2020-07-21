#!/usr/bin/env make

.PHONY: train docker-build docker-console spacy-load-model

DOCKER_IMAGE_VERSION=0.4.1
DOCKER_IMAGE_TAG=leovs09/sentiment:$(DOCKER_IMAGE_VERSION)

# ---------------------------------------------------------------------------------------------------------------------
# DEVELOPMENT
# ---------------------------------------------------------------------------------------------------------------------

train:
	python ./src/train.py

predict:
	python ./src/predict.py

test:
	python ./src/test.py

# Will load medium size model for spacy
spacy-load-md:
	python -m spacy download en_core_web_md

# ---------------------------------------------------------------------------------------------------------------------
# DOCKER
# ---------------------------------------------------------------------------------------------------------------------

# Will build docker image for development
docker-build:
	docker build --tag $(DOCKER_IMAGE_TAG) .

# Will start in docker develoment environment
docker-console:
	docker run -it --rm -v ${PWD}:/work -w /work $(DOCKER_IMAGE_TAG) bash


