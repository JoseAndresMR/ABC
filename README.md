# ABC

## Docker container

### Create container with GPU capabilities
You need to install in your own host previously [CUDA drivers](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=deb_local). In Ubuntu, you can check the installation by running `nvidia-smi`.

```
docker build --pull --rm -f "Dockerfile" -t abc:1.0 "."
```

### Run container

```
docker run -it --rm --gpus all --privileged -v $(pwd):/repo abc:1.0 bash
```
