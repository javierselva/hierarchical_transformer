#!/bin/bash
# $1 expects gpu device id (int)
# $2 expects custom config file (str) to replace default values in ./config
#        variable "model_name" should exist within the passed config file

# Add --user $(id-u) to run as specific user, Need to build docker such that it knows that user!!
# TODO add docker name for tracking!!
docker run --rm --gpus all --shm-size=16gb \
        -v /data-fast/128-data1/jselva/ssv2/:/data \
        -v /data-net/jselva/models/:/output \
        -v /home-net/jselva/hierarchical_transformer/:/workspace \
        -v /tmp/:/tmp \
        jselva/pytorch_new $1 $2    2>&1 | tee ../logs/$(grep "model_name" ../config/$2 | rev | cut -d '"' -f 2 | rev).log