.PHONY: help install train run clean test lint format

help:
	@echo "California Housing Price Prediction - Available Commands"
	@echo ""
	@echo "Usage: make [command]"
	@echo ""
	@echo "Commands:"
	@echo "  install       Install project dependencies"
	@echo "  train         Train the machine learning model"
	@echo "  run           Start the Streamlit application"
	@echo "  dev           Install dev dependencies"
	@echo "  lint          Run code quality checks (flake8, mypy)"
	@echo "  format        Format code with black and isort"
	@echo "  clean         Remove generated files and caches"
	@echo "  test          Run pytest tests"
	@echo "  help          Show this help message"

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt
	pip install -e .[dev]

train:
	python src/train.py

run:
	streamlit run src/main.py

lint:
	flake8 src --max-line-length=100
	mypy src

format:
	black src --line-length=100
	isort src

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type d -name .pytest_cache -exec rm -r {} +
	find . -type d -name .mypy_cache -exec rm -r {} +
	find . -type d -name .coverage -exec rm -r {} +
	find . -type d -name htmlcov -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/

test:
	pytest -v --cov=src tests/

all: install train run
