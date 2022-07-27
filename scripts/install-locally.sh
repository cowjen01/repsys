#!/bin/bash

./scripts/frontend-build.sh

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade --no-cache-dir -e .
