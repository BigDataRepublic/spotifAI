.PHONY: lint black

## Lint using flake8
lint:
	pflake8 spotifAI

## Format your code using black
black:
	python -m black --version
	python -m black spotifAI/data
	python -m black spotifAI/models
	python -m black spotifAI/deployment
	python -m black spotifAI/main.py

## Run static type checker for Python
mypy:
	mypy --version
	mypy spotifAI
