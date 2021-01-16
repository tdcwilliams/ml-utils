FROM ubuntu:bionic as base
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=$PYTHONPATH:/ml-utils \
    PYTHON_UNBUFFERED=1

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
      libgeos-dev \
      libglu1-mesa  \
      libglu1-mesa-dev \
      libgtk-3-dev \
      libjpeg-dev \
      libhdf5-serial-dev \
      liblapack-dev \
      libopenblas-dev \
      libpng-dev \
      libproj-dev \
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

# Non deep learning packages - install with conda
FROM base as simple
ENV PATH /opt/conda/bin:$PATH
COPY conda/requirements.txt /tmp/requirements.txt
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
 && rm /tmp/miniconda.sh \
 && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
 && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
 && echo "conda activate base" >> ~/.bashrc \
 && conda update -yq conda \
 && conda install -y -c conda-forge --file /tmp/requirements.txt \
 && /opt/conda/bin/conda clean -a \
 && rm -rf $HOME/.cache/yarn \
 && rm -rf /opt/conda/pkgs/*

# Deep learning packages - use pip since tensorflow v 2.0 is not in conda-forge
FROM base as deep
WORKDIR /tmp
RUN wget https://bootstrap.pypa.io/get-pip.py \
&&  python3 get-pip.py \
&&  rm get-pip.py \
&&  pip install --upgrade pip
RUN pip install \
      beautifulsoup4 \
      graphviz \
      imutils \
      jedi=0.17.2 \
      matplotlib \
      mock \
      nose \
      numpy \
      opencv-contrib-python \
      pandas \
      pillow \
      progressbar2 \
      proj \
      pydotplus \
      pyproj \
      pyyaml \
      scikit-image \
      scikit-learn \
      seaborn \
      shapely \
      statsmodels \
      tensorflow==2.0.0
RUN pip install \
      cartopy \
      cmocean \
      ipython \
      jupyter \
      nc-time-axis \
      netCDF4 \
      shapely --no-binary shapely \
      xgboost

WORKDIR /root
