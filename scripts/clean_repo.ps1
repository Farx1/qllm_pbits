# Clean Python artifacts from repository (Windows PowerShell)

Write-Host "Cleaning Python artifacts..." -ForegroundColor Cyan

# Remove __pycache__ directories
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

# Remove .pyc files
Get-ChildItem -Path . -Recurse -File -Filter "*.pyc" -ErrorAction SilentlyContinue | Remove-Item -Force

# Remove .pyo files
Get-ChildItem -Path . -Recurse -File -Filter "*.pyo" -ErrorAction SilentlyContinue | Remove-Item -Force

# Remove pytest cache
if (Test-Path .pytest_cache) { Remove-Item -Recurse -Force .pytest_cache }

# Remove coverage files
if (Test-Path .coverage) { Remove-Item -Force .coverage }
if (Test-Path htmlcov) { Remove-Item -Recurse -Force htmlcov }

# Remove build artifacts
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
Get-ChildItem -Path . -Filter "*.egg-info" -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

Write-Host "✓ Cleanup complete" -ForegroundColor Green

