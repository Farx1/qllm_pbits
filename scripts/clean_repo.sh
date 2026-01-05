#!/bin/bash
# Clean Python artifacts from repository

echo "Cleaning Python artifacts..."

# Remove __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove .pyc files
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Remove .pyo files
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove pytest cache
rm -rf .pytest_cache 2>/dev/null || true

# Remove coverage files
rm -f .coverage 2>/dev/null || true
rm -rf htmlcov 2>/dev/null || true

# Remove build artifacts
rm -rf build dist *.egg-info 2>/dev/null || true

echo "✓ Cleanup complete"

