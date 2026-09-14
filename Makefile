.DEFAULT_GOAL := help
.PHONY: help setup lint format test test-integration test-data up down logs data eda baseline train experiment-leakage

UV      ?= uv
RUN     := $(UV) run
COMPOSE ?= docker compose

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- development
setup: ## Create the Python 3.12 environment and install git hooks
	$(UV) sync --extra data
	$(RUN) pre-commit install

lint: ## Ruff lint + format check + mypy (strict)
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests
	$(RUN) mypy

format: ## Auto-format and auto-fix lint issues
	$(RUN) ruff format src tests
	$(RUN) ruff check --fix src tests

test: ## Unit tests (no services, no dataset)
	$(RUN) pytest

test-integration: ## Tests that need `make up` (Redis, Redpanda)
	$(RUN) pytest -m integration

test-data: ## Tests that need the prepared IEEE-CIS dataset
	$(RUN) pytest -m data

# ---------------------------------------------------------------- services
up: ## Start infrastructure (Redpanda, Redis, MLflow) and wait for health
	$(COMPOSE) up -d --wait

down: ## Stop all services
	$(COMPOSE) down

logs: ## Tail service logs
	$(COMPOSE) logs -f --tail=100

# ---------------------------------------------------------------- phase 0
data: ## Download IEEE-CIS (needs Kaggle token) and build the canonical event table
	$(RUN) bastion data download
	$(RUN) bastion data prepare

eda: ## Generate the EDA report into docs/results/phase0/
	$(RUN) bastion eda

baseline: ## Evaluate the rules baseline into docs/results/phase0/
	$(RUN) bastion baseline rules

# ---------------------------------------------------------------- phase 1
train: ## Train and calibrate LightGBM on point-in-time features (logged to MLflow)
	$(RUN) bastion train

experiment-leakage: ## Leakage experiment: naive vs point-in-time features, shuffled vs temporal split
	$(RUN) bastion experiment leakage
