#!/bin/bash

./scripts/build-frontend.sh

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade --no-cache-dir -e .
