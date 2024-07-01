#!/usr/bin/env bash

REPO_PATH=$PWD

# Remove previous service build
if test -d learning_service; then
  echo "Removing previous service build"
  sudo rm -r learning_service
fi

# Push packages and fetch service
make clean

autonomy push-all

autonomy fetch --local --service valory/learning_service && cd learning_service

# Build the image
autonomy init --reset --author author --remote --ipfs --ipfs-node "/dns/registry.autonolas.tech/tcp/443/https"
autonomy build-image

# Copy .env file
cp $REPO_PATH/.env .

# Copy the keys and build the deployment
cp $REPO_PATH/keys.json .

autonomy deploy build -ltm

# Run the deployment
autonomy deploy run --build-dir abci_build/
