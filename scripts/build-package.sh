#!/bin/bash

./scripts/build-frontend.sh

pip install -e ".[dev]"
python -m build
