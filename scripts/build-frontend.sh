#!/bin/bash

if [ -d "./repsys/web/build" ]; then
  rm -rf ./repsys/web/build
fi

cd ./frontend || exit
npm install
npm run build

cd ..
mkdir -p ./repsys/web
cp -r ./frontend/build ./repsys/web/build
