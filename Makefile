.PHONY: clean install test test-all demo help

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

clean:  ## Remove Python artifacts (__pycache__, .pyc, .pytest_cache, etc.)
	@echo "Cleaning Python artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@rm -rf .pytest_cache .coverage htmlcov build dist *.egg-info 2>/dev/null || true
	@echo "✓ Cleanup complete"

install:  ## Install package in editable mode with dev dependencies
	pip install -e ".[dev]"

test:  ## Run fast tests only (no HuggingFace model downloads)
	pytest -m "not slow" -v

test-all:  ## Run all tests including slow HuggingFace integration tests
	RUN_SLOW=1 pytest -v

demo:  ## Run demonstration script
	python demo_sampler.py

