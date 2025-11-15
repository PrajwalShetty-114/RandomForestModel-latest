#!/usr/bin/env bash
set -euo pipefail

echo "Upgrade pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "Install core numeric wheels (prefer binary wheels)..."
python -m pip install --prefer-binary numpy==1.24.4 scipy==1.10.1

echo "Install remaining requirements..."
python -m pip install --prefer-binary -r requirements.txt

echo "Build script finished."
