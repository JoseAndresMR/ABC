FROM nvidia/cuda:11.0-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install Python 3.7
RUN apt-get -y update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y update && apt-get -y install python3.7 \
    python3-pip \
    swig \
    python3-opengl

# Install Python libraries with pip
RUN python3.7 -m pip install -U --no-cache-dir pip && python3.7 -m \
    pip install --no-cache-dir \
    cloudpickle==2.0.0 \
    gym==0.21.0 \
    # numpy==1.22.2 \
    pyglet==1.5.21

VOLUME /repo
WORKDIR /repo
CMD [ "python3.7", "gym_render.py"]