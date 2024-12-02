FROM nvcr.io/nvidia/pytorch:23.02-py3

# TODO after adding decord flash fails to import. Leaving it out for now as flash is only needed for LARS
RUN pip install av einops h5py matplotlib pytorch-ignite pytorch-lightning  \
    lightning-flash wandb umap-learn tsnecuda scikit-learn decord

# Alternatively to compile the decord from source
#ENV DEBIAN_FRONTEND=noninteractive
#ENV TZ=Eurpe/Madrid
#RUN apt-get update
#RUN apt-get install -y software-properties-common
#RUN add-apt-repository ppa:savoury1/ffmpeg4
#RUN apt-get update
#RUN apt-get install -y build-essential python3-dev python3-setuptools make cmake
#RUN apt-get install -y ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
#
#RUN git clone --recursive https://github.com/dmlc/decord /root/decord
#RUN cd /root/decord && mkdir build && cd build && cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
#RUN cd ../python && python3 setup.py install
#
## Test decord was installed by opening python and importing decord
#RUN python3 -c "import decord"

ENTRYPOINT ["/bin/bash", "launcher.sh"]