#!/bin/bash

./scripts/build-frontend.sh

pip install --upgrade pip
pip install --upgrade build setuptools wheel

python -m build
