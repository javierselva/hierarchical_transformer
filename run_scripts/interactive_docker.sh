#!/bin/bash

# Add --user $(id-u) to run as specific user, Need to build docker such that it knows that user!!
docker run --rm --gpus all -it --entrypoint /bin/bash --shm-size=3gb \
        -v /data-local/data1-ssd/jselva/ucf101/:/data \
        -v /data-net/jselva/models/:/output \
        -v /home-net/jselva/hierarchical_transformer/:/workspace \
        jselva/pytorch_new