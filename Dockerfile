FROM nvidia/cuda:11.0-base-ubuntu18.04

# Environments
ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
CMD nvidia-smi

# Install linux CUDA 11.0
RUN apt-get update -y && apt-get install -y cuda-nvcc-11-0

# Install Python 3.7
RUN apt-get -y update && apt-get -y install \
    python3.7 \
    python3-pip \
    swig

# Install Python libraries with pip
RUN python3.7 -m pip install -U --no-cache-dir pip && python3.7 -m \
    pip install --no-cache-dir \
    absl-py==1.0.0 \
    ale-py==0.7.3 \
    alembic==1.7.5 \
    argcomplete==1.12.3 \
    argon2-cffi==21.1.0 \
    astunparse==1.6.3 \
    attrs==21.2.0 \
    autopage==0.4.0 \
    backcall==0.2.0 \
    bleach==4.1.0 \
    box2d-py==2.3.8 \
    cached-property==1.5.2 \
    certifi==2021.10.8 \
    cffi==1.15.0 \
    charset-normalizer==2.0.7 \
    cliff==3.10.0 \
    cloudpickle==2.0.0 \
    cmaes==0.8.2 \
    cmd2==2.3.2 \
    cycler==0.11.0 \
    debugpy==1.5.1 \
    decorator==5.1.0 \
    defusedxml==0.7.1 \
    docopt==0.6.2 \
    entrypoints==0.3 \
    flatbuffers==2.0 \
    gast==0.4.0 \
    google-auth==2.3.3 \
    google-auth-oauthlib==0.4.6 \
    google-pasta==0.2.0 \
    greenlet==1.1.2 \
    grpcio==1.41.1 \
    gym==0.21.0 \
    h5py==3.5.0 \
    idna==3.3 \
    importlib-metadata==4.8.2 \
    importlib-resources==5.4.0 \
    iniconfig==1.1.1 \
    ipykernel==6.5.0 \
    ipython==7.29.0 \
    ipython-genutils==0.2.0 \
    ipywidgets==7.6.5 \
    jedi==0.18.0 \
    Jinja2==3.0.3 \
    jsonschema==4.2.1 \
    jupyter==1.0.0 \
    jupyter-client==7.0.6 \
    jupyter-console==6.4.0 \
    jupyter-core==4.9.1 \
    jupyterlab-pygments==0.1.2 \
    jupyterlab-widgets==1.0.2 \
    keras==2.7.0 \
    Keras-Preprocessing==1.1.2 \
    kiwisolver==1.3.2 \
    libclang==12.0.0 \
    Mako==1.1.6 \
    Markdown==3.3.4 \
    MarkupSafe==2.0.1 \
    matplotlib==3.4.3 \
    matplotlib-inline==0.1.3 \
    mistune==0.8.4 \
    mlagents-envs==0.27.0 \
    mock==4.0.3 \
    nbclient==0.5.8 \
    nbconvert==6.3.0 \
    nbformat==5.1.3 \
    nest-asyncio==1.5.1 \
    notebook==6.4.5 \
    numpy==1.21.4 \
    oauthlib==3.1.1 \
    opt-einsum==3.3.0 \
    optuna==2.10.0 \
    packaging==21.2 \
    pandas==1.3.4 \
    pandocfilters==1.5.0 \
    parso==0.8.2 \
    pbr==5.8.0 \
    pdoc==8.0.1 \
    pexpect==4.8.0 \
    pickleshare==0.7.5 \
    Pillow==8.4.0 \
    pluggy==1.0.0 \
    prettytable==2.4.0 \
    prometheus-client==0.12.0 \
    prompt-toolkit==3.0.22 \
    protobuf==3.19.1 \
    ptyprocess==0.7.0 \
    py==1.11.0 \
    pyasn1==0.4.8 \
    pyasn1-modules==0.2.8 \
    pycparser==2.21 \
    pyglet==1.5.21 \
    Pygments==2.10.0 \
    pyparsing==2.4.7 \
    pyperclip==1.8.2 \
    pyrsistent==0.18.0 \
    pytest==6.2.5 \
    python-dateutil==2.8.2 \
    pytz==2021.3 \
    PyYAML==6.0 \
    pyzmq==22.3.0 \
    qtconsole==5.2.0 \
    QtPy==1.11.2 \
    requests==2.26.0 \
    requests-oauthlib==1.3.0 \
    rsa==4.7.2 \
    scipy==1.7.2 \
    seaborn==0.11.2 \
    Send2Trash==1.8.0 \
    six==1.16.0 \
    SQLAlchemy==1.4.27 \
    stevedore==3.5.0 \
    tensorboard==2.7.0 \
    tensorboard-data-server==0.6.1 \
    tensorboard-plugin-wit==1.8.0 \
    tensorflow==2.7.0 \
    tensorflow-estimator==2.7.0 \
    tensorflow-io-gcs-filesystem==0.22.0 \
    termcolor==1.1.0 \
    terminado==0.12.1 \
    testpath==0.5.0 \
    toml==0.10.2 \
    tornado==6.1 \
    tqdm==4.62.3 \
    traitlets==5.1.1 \
    typing_extensions==4.0.0 \
    urllib3==1.26.7 \
    wcwidth==0.2.5 \
    webencodings==0.5.1 \
    Werkzeug==2.0.2 \
    widgetsnbextension==3.5.2 \
    wrapt==1.13.3 \
    zipp==3.6.0 \
    setuptools==59.5.0

# # Install Torch with CUDA. Tensorflow apparently was not needed
# RUN pip3 install tensorflow==2.0.0
RUN python3.7 -m pip install -U --no-cache-dir pip && python3.7 -m \
    pip install --no-cache-dir \
    torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html


# Install library
ADD . /repo/
WORKDIR /repo/
RUN python3.7 -m pip install -e .

VOLUME /repo
WORKDIR /repo
