#!/usr/bin/env make

.PHONY: train docker-build docker-console spacy-load-model

# ---------------------------------------------------------------------------------------------------------------------
# DEVELOPMENT
# ---------------------------------------------------------------------------------------------------------------------

train:
	python ./src/index.py

# Will load medium size model for spacy
spacy-load-md:
	python -m spacy download en_core_web_md

# ---------------------------------------------------------------------------------------------------------------------
# DOCKER
# ---------------------------------------------------------------------------------------------------------------------

# Will build docker image for development
docker-build:
	docker build --tag nlp:0.3.0-py .

# Will start in docker develoment environment
docker-console:
	docker run -it --rm -v ${PWD}:/work -w /work nlp:0.3.0-py bash


