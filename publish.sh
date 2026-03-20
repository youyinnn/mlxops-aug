#!/bin/bash
set -e

# Publish mlxops-aug to PyPI
# Usage: ./publish.sh [--test]   (--test publishes to TestPyPI)

PYPI_REPO="pypi"
if [[ "$1" == "--test" ]]; then
    PYPI_REPO="testpypi"
    echo "Publishing to TestPyPI..."
else
    echo "Publishing to PyPI..."
fi

# Clean previous builds
rm -rf dist/ build/

# Build the package
echo "Building package..."
python -m build

# Upload
echo "Uploading to $PYPI_REPO..."
python -m twine upload --repository "$PYPI_REPO" dist/*

echo "Done."
