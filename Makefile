#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = gpt-2
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
DOCKER_BUILDKIT=1

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Download and process fineweb dataset
.PHONY: download-fineweb
download-fineweb:
	python -m gpt.fineweb.dataset

## Cache hellaswag dataset
.PHONY: cache-hellaswag
cache-hellaswag:
	python -m gpt.hellaswag.dataset

## Train the model
.PHONY: train
train:
	torchrun --nproc_per_node=1 -m gpt.train

## Run tests
.PHONY: test
test:
	python -m pytest tests

#################################################################################
# DOCKER                                                                 #
#################################################################################

## Build Docker image
.PHONY: docker-build
docker-build:
	docker compose build

## Run tests in Docker
.PHONY: docker-test
docker-test:
	docker compose up base

## Run model training in Docker
.PHONY: docker-train
docker-train:
	docker compose up train

## Debug Docker container with interactive shell
.PHONY: docker-debug
docker-debug:
	docker compose run --rm train /bin/bash


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
