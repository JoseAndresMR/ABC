# Project Title

Haru Wizard of Oz project.

## Description


# Getting Started


## Prerequisites

Ubuntu 18.04 with ROS Melodic

    $ sudo apt install python-rosinstall python-rosinstall-generator python-wstool build-essential python3 python3-pip

    $ sudo apt install build-essential virtualenv

### Dependencies

- clone the repository

    $ git clone https://joseandres4i@bitbucket.org/gperez9/abc.git

- then setup a virtualenv:

    $ cd abc
    $ python3 -m venv venv_abc
    $ source venv_abc/bin/activate
    $ pip install --upgrade pip wheel setuptools

- install requirements:

    $ pip install -r requirements.txt --no-cache-dir
    $ deactivate

### Executing program

Terminal 1:

    $ source venv_abc/bin/activate
    $ python src/management/Testbench.py

Terminal 2:

    $ tensorboard --logdir data/runs/biwalker_tests

Browser:

    $ http://localhost:6006/

### Architecture

![ABC scheme](./bin/pcitures/ABC scheme.png)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Contact

Project Link: [https://github.com/JoseAndresMR/ABC](https://github.com/JoseAndresMR/ABC)

# Docker container

## Create container with GPU capabilities
You need to install in your own host previously [CUDA drivers](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=deb_local). In Ubuntu, you can check the installation by running `nvidia-smi`.

```
docker build --pull --rm -f "Dockerfile" -t abc:1.0 "."
```

## Run container

```
docker run -it --rm --gpus all --privileged -v $(pwd):/repo abc:1.0 bash
```

## Run container with X11 capabilities

In Ubuntu, first you have to prepare `xhost`:

```
xhost +
```

Later, you should run the docker with this arguments:

```
docker run -it --rm --gpus all --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/repo abc:1.0 bash
```

After you finish of using the docker, it is a good practice to close `xhost`:

```
xhost -
```