FROM nvidia/cuda:11.0-base-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install Python 3.8
RUN apt-get -y update && apt-get -y install \
    python3.8 \
    python3-pip \
    swig \
    python3-opengl

# Install Python libraries with pip
RUN python3.8 -m pip install -U --no-cache-dir pip && python3.8 -m \
    pip install --no-cache-dir \
    cloudpickle==2.0.0 \
    gym==0.21.0 \
    # numpy==1.22.2 \
    pyglet==1.5.21

VOLUME /repo
WORKDIR /repo
CMD [ "python3.8", "gym_render.py"]
