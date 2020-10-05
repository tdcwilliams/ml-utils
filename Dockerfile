FROM ubuntu:bionic
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=$PYTHONPATH:/ml-utils

RUN apt-get update \
&&  apt-get upgrade -y \
&&  apt-get install -y \
      build-essential \
      cmake \
      g++-6 \
      gcc-6 \
      gfortran \
      graphviz \
      libatlas-base-dev \
      libavcodec-dev \
      libavformat-dev \
      libglu1-mesa  \
      libglu1-mesa-dev \
      libgtk-3-dev \
      libjpeg-dev \
      libhdf5-serial-dev \
      liblapack-dev \
      libopenblas-dev \
      libpng-dev \
      libswscale-dev \
      libtiff-dev \
      libv4l-dev \
      libx264-dev \
      libxi-dev  \
      libxmu-dev \
      libxvidcore-dev \
      pkg-config \
      python3-dev \
      python3-tk \
      python-imaging-tk \
      unzip \
      wget \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN wget https://bootstrap.pypa.io/get-pip.py \
&&  python3 get-pip.py \
&&  rm get-pip.py \
&&  pip install --upgrade pip \
&&  pip install \
      beautifulsoup4 \
      dlib \
      graphviz \
      imutils \
      ipython \
      jupyter \
      matplotlib \
      mock \
      nose \
      numpy \
      opencv-contrib-python \
      pandas \
      pillow \
      progressbar2 \
      pydotplus \
      pyyaml \
      scikit-image \
      scikit-learn \
      seaborn \
      shapely \
      statsmodels \
      tensorflow==2.0.0

WORKDIR /root
