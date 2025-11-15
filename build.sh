#!/usr/bin/env bash
set -euo pipefail

echo "Upgrade pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "Remove any existing numpy (clean slate)..."
python -m pip uninstall -y numpy || true

echo "Install a fresh binary numpy wheel (prefer-binary + force reinstall)..."
# pick a stable release; if this errors we can try 1.24.x or 1.26.x
python -m pip install --prefer-binary --upgrade --force-reinstall numpy==1.25.2

echo "Install binary scipy (if needed)..."
python -m pip install --prefer-binary --upgrade --force-reinstall scipy==1.10.1

echo "Install remaining requirements (prefer binary wheels)..."
python -m pip install --prefer-binary -r requirements.txt

echo "Sanity check: show numpy and sklearn versions"
python -c "import numpy, sklearn; print('numpy', numpy.__version__, numpy.__file__); print('sklearn', sklearn.__version__)"
