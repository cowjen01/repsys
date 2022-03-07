#!/bin/bash

./scripts/frontend-build.sh

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build setuptools wheel

python3 -m build
