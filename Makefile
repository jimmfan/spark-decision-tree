PROJECT=learn-project-infra-dev
VERSION=0.1.0
PYTHON_VERSION=3.10
SOURCE_OBJECTS=src tests

setup:
	pip3 install poetry==1.3.1
	poetry install --no-root

format.black:
	poetry run black ${SOURCE_OBJECTS}
format.isort:
	poetry run isort --atomic ${SOURCE_OBJECTS}
format: format.isort format.black

lints.format.check:
	poetry run black --check ${SOURCE_OBJECTS}
	poetry run isort --check-only ${SOURCE_OBJECTS}

test: setup
	poetry run coverage run -m pytest -s .

test.coverage: test
	poetry run coverage report -m --fail-under=90