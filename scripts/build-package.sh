#!/bin/bash

if [ -d "./repsys/web/build" ]; then
  rm -rf ./repsys/web/build
fi

./scripts/build-frontend.sh

pip install -e ".[dev]"
python -m build
