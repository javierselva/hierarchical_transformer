#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=$1 HDF5_USE_FILE_LOCKING=FALSE python main.py $2