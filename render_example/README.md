# Example of render

To test how we can obtain the render of an docker, here three test are presented:
- Python3.9 (it works)
- Python3.8 (it does not work)
- Python3.7 (it does not work)

`brainrl` library is made in Python3.7, so it may indicate that we should move to python3.9.


## Build dataset

Version available is `python3.7`, `python3.8`, and `python3.9`.

```
docker build --rm -f gym38.Dockerfile -t gym:3.9 .
```

## Run dataset

In Ubuntu, first you have to prepare `xhost`:

```
xhost +
```


Then, you can run it the version of the docker.

```
docker run -it --rm --gpus all --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/repo gym:3.9
```

After you finish of using the docker, it is a good practice to close `xhost`:

```
xhost -
```
