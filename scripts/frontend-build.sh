#!/bin/bash

cd ./frontend || exit
npm install
npm run build

cd ..
./scripts/copy-frontend.sh
