#!/bin/bash

if [ -d "./repsys/web/build" ]; then
  rm -rf ./repsys/web/build
fi

mkdir -p ./repsys/web
cp -r ./frontend/build ./repsys/web/build
