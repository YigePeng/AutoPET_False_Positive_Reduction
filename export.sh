#!/usr/bin/env bash

./build.sh

docker save hybrid_cnn | gzip -c > hybrid_cnn.tar.gz
