.PHONY: lint black

## Lint using flake8
lint:
	pflake8 --version
	pflake8 spotifAI

## Format your code using black
black:
	python -m black --version
	python -m black spotifAI

## Run static type checker for Python
mypy:
	mypy --version
	mypy spotifAI
