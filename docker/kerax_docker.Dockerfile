ARG cuda_major=10
ARG cuda_minor=0
# cuda_variant: (base|runtime|devel)
ARG cuda_variant=runtime
ARG ubuntu_version=18.04
FROM nvidia/cuda:${cuda_major}.${cuda_minor}-cudnn7-${cuda_variant}-ubuntu${ubuntu_version}


RUN apt-get update \
  && apt-get install --yes --no-install-recommends \
  build-essential \
  bzip2 \
  curl \
  git \
  wget \
  zip \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# https://riptutorial.com/tensorflow/example/13427/use-the-tcmalloc-allocator
RUN apt-get update \
  && apt-get install -y --no-install-recommends libgoogle-perftools4 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

# Instal requirements for kerax.
RUN python3 -m pip --no-cache-dir install --upgrade \
    albumentations==0.3.2 \
    Keras==2.2.4 \
    keras-tqdm==2.0.1 \
    scikit-learn==0.23.1 \
    tensorflow-gpu==1.15.0

RUN pip install --no-cache-dir --no-deps --upgrade \
  git+https://www.github.com/keras-team/keras-contrib.git
