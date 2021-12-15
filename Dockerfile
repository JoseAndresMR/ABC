FROM nvidia/cuda:11.4.1-base-ubuntu18.04

# Environments
ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# CMD nvidia-smi

# # Install linux CUDA 11.0
RUN apt-get update -y && apt-get install -y cuda-nvcc-11-4

# Install Python 3.7
RUN apt-get -y update && apt-get -y install \
    python3.7
RUN apt-get -y update && apt-get -y install \
    python3-pip
    #ipython3
RUN apt-get -y update && apt-get -y install swig

# RUN python3.7 -m pip install -U pip && python3.7 -m pip install numpy==1.18.5 \
#     box2d-py

# Install libraries
ADD . /repo/
WORKDIR /repo/
RUN python3.7 -m pip install -U pip && python3.7 -m \
    pip install -r requirements.txt
RUN python3.7 -m pip install -e .

VOLUME /repo
WORKDIR /repo
