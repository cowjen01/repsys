#!/bin/bash

if [ ! -d "./repsys/web/build" ]; then
  ./scripts/build-frontend.sh
fi

pip install -e .
