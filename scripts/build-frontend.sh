#!/bin/bash

cd ./frontend || exit
npm install
npm run build

cd ..
mkdir -p ./repsys/web
cp -r ./frontend/build ./repsys/web/build
