# NVIDIA CUDA

FROM nvidia/cuda:10.1-cudnn7-devel

# Python 3.6
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.6 python3.6-dev python3-pip wget git sudo nano && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://conda.anaconda.org/conda-forge/linux-64/ca-certificates-2018.4.16-0.tar.bz2 && \
    tar -xjf ca-certificates-2018.4.16-0.tar.bz2 -C /usr/bin && \
    rm ca-certificates-2018.4.16-0.tar.bz2

RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3 && ln -sfn /usr/bin/python3 /usr/bin/python && ln -sfn /usr/bin/pip3 /usr/bin/pip

# create a root user
#ARG USER_ID=1000
#RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
#RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER root
WORKDIR /home/root

ENV PATH="/home/root/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install dependencies and packages
RUN apt-get update && apt-get install -y \
    libsm6 libxrender1 libfontconfig1 python3.6-tk && \
    apt install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir python-dateutil==2.7.2 && \
    pip install --no-cache-dir numpy==1.14.3 && \
    pip install --no-cache-dir scipy==1.1.0 && \
    pip install --no-cache-dir seaborn && \
    pip install --no-cache-dir imageio && \
    pip install --no-cache-dir opencv-contrib-python && \
    pip install --no-cache-dir bz2file && \
    pip install --no-cache-dir certifi==2018.4.16 && \
    pip install --no-cache-dir tqdm==4.23.0 && \
    pip install --no-cache-dir wheel==0.31.0 && \
    pip install --no-cache-dir torch==1.1.0 torchvision==0.3.0 && \
    pip install --no-cache-dir tensorboard==1.14.0 && \
    pip install --no-cache-dir tensorboardX==2.0 && \
    pip install --no-cache-dir lmdb && \
    pip install --no-cache-dir pillow && \
    pip install --no-cache-dir nltk && \
    pip install --no-cache-dir natsort


    # Torch 0.4.1
#RUN sudo wget https://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-linux_x86_64.whl && \
#    sudo pip install torch-0.4.1-cp36-cp36m-linux_x86_64.whl && \
#    sudo rm torch-0.4.1-cp36-cp36m-linux_x86_64.whl

    # Pytorch select 0.1 cpu
RUN sudo wget https://conda.anaconda.org/anaconda/linux-64/_pytorch_select-0.1-cpu_0.tar.bz2 && \
    sudo tar -xjf _pytorch_select-0.1-cpu_0.tar.bz2 -C /usr/bin && \
    sudo rm _pytorch_select-0.1-cpu_0.tar.bz2

    # Numpy-base 1.14.3
RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/numpy-base-1.14.3-py36h9be14a7_1.tar.bz2 && \
    sudo tar -xjf numpy-base-1.14.3-py36h9be14a7_1.tar.bz2 -C /usr/bin && \
    sudo rm numpy-base-1.14.3-py36h9be14a7_1.tar.bz2

    # Cudatoolkit 10.2
#RUN sudo wget https://conda.anaconda.org/anaconda/linux-64/cudatoolkit-10.2.89-hfd86e86_0.tar.bz2 && \
#    sudo tar -xjf cudatoolkit-10.2.89-hfd86e86_0.tar.bz2 -C /usr/bin && \
#    sudo rm cudatoolkit-10.2.89-hfd86e86_0.tar.bz2

    # Cudatoolkit 10.1
RUN sudo wget https://anaconda.org/anaconda/cudatoolkit/10.1.243/download/linux-64/cudatoolkit-10.1.243-h6bb024c_0.tar.bz2 && \
    sudo tar -xjf cudatoolkit-10.1.243-h6bb024c_0.tar.bz2 -C /usr/bin && \
    sudo rm cudatoolkit-10.1.243-h6bb024c_0.tar.bz2

#     # Blas 1.0
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/blas-1.0-mkl.tar.bz2 && \
#     sudo tar -xjf blas-1.0-mkl.tar.bz2 -C /usr/bin && \
#     sudo rm blas-1.0-mkl.tar.bz2

#     # Cairo 1.14.12
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/cairo-1.14.12-h7636065_2.tar.bz2 && \
#     sudo tar -xjf cairo-1.14.12-h7636065_2.tar.bz2 -C /usr/bin && \
#     sudo rm cairo-1.14.12-h7636065_2.tar.bz2

#     # Cudatoolkit 8.0.3
# #RUN sudo wget https://repo.continuum.io/pkgs/free/linux-64/cudatoolkit-8.0-3.tar.bz2 && \
# #    sudo tar -xjf cudatoolkit-8.0-3.tar.bz2 -C /usr/bin && \
# #    sudo rm cudatoolkit-8.0-3.tar.bz2

#     # Dbus 1.13.2
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/dbus-1.13.2-h714fa37_1.tar.bz2 && \
#     sudo tar -xjf dbus-1.13.2-h714fa37_1.tar.bz2 -C /usr/bin && \
#     sudo rm dbus-1.13.2-h714fa37_1.tar.bz2

#     # Expat 2.2.5
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/expat-2.2.5-he0dffb1_0.tar.bz2 && \
#     sudo tar -xjf expat-2.2.5-he0dffb1_0.tar.bz2 -C /usr/bin && \
#     sudo rm expat-2.2.5-he0dffb1_0.tar.bz2

#     # Fotnconfig 2.12.6
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/fontconfig-2.12.6-h49f89f6_0.tar.bz2 && \
#     sudo tar -xjf fontconfig-2.12.6-h49f89f6_0.tar.bz2 -C /usr/bin && \
#     sudo rm fontconfig-2.12.6-h49f89f6_0.tar.bz2

#     # Freeglut 2.8.1
# RUN sudo wget https://repo.continuum.io/pkgs/free/linux-64/freeglut-2.8.1-0.tar.bz2 && \
#     sudo tar -xjf freeglut-2.8.1-0.tar.bz2 -C /usr/bin && \
#     sudo rm freeglut-2.8.1-0.tar.bz2

#     # Freetype 2.8
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/freetype-2.8-hab7d2ae_1.tar.bz2 && \
#     sudo tar -xjf freetype-2.8-hab7d2ae_1.tar.bz2 -C /usr/bin && \
#     sudo rm freetype-2.8-hab7d2ae_1.tar.bz2

#     # Gflags 2.2.1
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/gflags-2.2.1-hf484d3e_0.tar.bz2 && \
#     sudo tar -xjf gflags-2.2.1-hf484d3e_0.tar.bz2 -C /usr/bin && \
#     sudo rm gflags-2.2.1-hf484d3e_0.tar.bz2

#     # Glib 2.56.1
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/glib-2.56.1-h000015b_0.tar.bz2 && \
#     sudo tar -xjf glib-2.56.1-h000015b_0.tar.bz2 -C /usr/bin && \
#     sudo rm glib-2.56.1-h000015b_0.tar.bz2

#     # Glog 0.3.5
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/glog-0.3.5-hf484d3e_1.tar.bz2 && \
#     sudo tar -xjf glog-0.3.5-hf484d3e_1.tar.bz2 -C /usr/bin && \
#     sudo rm glog-0.3.5-hf484d3e_1.tar.bz2

#     # Graphite2 1.3.11
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/graphite2-1.3.11-hf63cedd_1.tar.bz2 && \
#     sudo tar -xjf graphite2-1.3.11-hf63cedd_1.tar.bz2 -C /usr/bin && \
#     sudo rm graphite2-1.3.11-hf63cedd_1.tar.bz2

#     # Gst-plugins-base 1.14.0
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/gst-plugins-base-1.14.0-hbbd80ab_1.tar.bz2 && \
#     sudo tar -xjf gst-plugins-base-1.14.0-hbbd80ab_1.tar.bz2 -C /usr/bin && \
#     sudo rm gst-plugins-base-1.14.0-hbbd80ab_1.tar.bz2

#     # Gstreamer 1.14.0
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/gstreamer-1.14.0-hb453b48_1.tar.bz2 && \
#     sudo tar -xjf gstreamer-1.14.0-hb453b48_1.tar.bz2 -C /usr/bin && \
#     sudo rm gstreamer-1.14.0-hb453b48_1.tar.bz2

#     # Harfbuzz 1.7.6
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/harfbuzz-1.7.6-h5f0a787_1.tar.bz2 && \
#     sudo tar -xjf harfbuzz-1.7.6-h5f0a787_1.tar.bz2 -C /usr/bin && \
#     sudo rm harfbuzz-1.7.6-h5f0a787_1.tar.bz2

#     # Hdf5 1.8.18
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/hdf5-1.8.18-h6792536_1.tar.bz2 && \
#     sudo tar -xjf hdf5-1.8.18-h6792536_1.tar.bz2 -C /usr/bin && \
#     sudo rm hdf5-1.8.18-h6792536_1.tar.bz2

#     # Icu 58.2
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/icu-58.2-h9c2bf20_1.tar.bz2 && \
#     sudo tar -xjf icu-58.2-h9c2bf20_1.tar.bz2 -C /usr/bin && \
#     sudo rm icu-58.2-h9c2bf20_1.tar.bz2

#     # Intel Openmp 2018.0.0-8
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/intel-openmp-2018.0.0-8.tar.bz2 && \
#     sudo tar -xjf intel-openmp-2018.0.0-8.tar.bz2 -C /usr/bin && \
#     sudo rm intel-openmp-2018.0.0-8.tar.bz2

#     # Jasper 2.0.14
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/jasper-2.0.14-h07fcdf6_0.tar.bz2 && \
#     sudo tar -xjf jasper-2.0.14-h07fcdf6_0.tar.bz2 -C /usr/bin && \
#     sudo rm jasper-2.0.14-h07fcdf6_0.tar.bz2

#     # Jpeg 9b
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/jpeg-9b-h024ee3a_2.tar.bz2 && \
#     sudo tar -xjf jpeg-9b-h024ee3a_2.tar.bz2 -C /usr/bin && \
#     sudo rm jpeg-9b-h024ee3a_2.tar.bz2

#     # Libedit 3.1
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/libedit-3.1-heed3624_0.tar.bz2 && \
#     sudo tar -xjf libedit-3.1-heed3624_0.tar.bz2 -C /usr/bin && \
#     sudo rm libedit-3.1-heed3624_0.tar.bz2

#     # Libffi 3.2.1
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/libffi-3.2.1-hd88cf55_4.tar.bz2 && \
#     sudo tar -xjf libffi-3.2.1-hd88cf55_4.tar.bz2 -C /usr/bin && \
#     sudo rm libffi-3.2.1-hd88cf55_4.tar.bz2

#     # Libgcc ng 7.2.0
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/libgcc-ng-7.2.0-hdf63c60_3.tar.bz2 && \
#     sudo tar -xjf libgcc-ng-7.2.0-hdf63c60_3.tar.bz2 -C /usr/bin && \
#     sudo rm libgcc-ng-7.2.0-hdf63c60_3.tar.bz2

#     # Libgfortran ng 7.2.0
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/libgfortran-ng-7.2.0-hdf63c60_3.tar.bz2 && \
#     sudo tar -xjf libgfortran-ng-7.2.0-hdf63c60_3.tar.bz2 -C /usr/bin && \
#     sudo rm libgfortran-ng-7.2.0-hdf63c60_3.tar.bz2

#     # Libglu 9.0.0
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/libglu-9.0.0-h0c0bdc1_1.tar.bz2 && \
#     sudo tar -xjf libglu-9.0.0-h0c0bdc1_1.tar.bz2 -C /usr/bin && \
#     sudo rm libglu-9.0.0-h0c0bdc1_1.tar.bz2

#     # Libopus 1.2.1
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/libopus-1.2.1-hb9ed12e_0.tar.bz2 && \
#     sudo tar -xjf libopus-1.2.1-hb9ed12e_0.tar.bz2 -C /usr/bin && \
#     sudo rm libopus-1.2.1-hb9ed12e_0.tar.bz2

#     # Libpng 1.6.34
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/libpng-1.6.34-hb9fc6fc_0.tar.bz2 && \
#     sudo tar -xjf libpng-1.6.34-hb9fc6fc_0.tar.bz2 -C /usr/bin && \
#     sudo rm libpng-1.6.34-hb9fc6fc_0.tar.bz2

#     # Libprotobuf 3.5.2
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/libprotobuf-3.5.2-h6f1eeef_0.tar.bz2 && \
#     sudo tar -xjf libprotobuf-3.5.2-h6f1eeef_0.tar.bz2 -C /usr/bin && \
#     sudo rm libprotobuf-3.5.2-h6f1eeef_0.tar.bz2

#     # Libstdcxx ng 7.2.0
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/libstdcxx-ng-7.2.0-hdf63c60_3.tar.bz2 && \
#     sudo tar -xjf libstdcxx-ng-7.2.0-hdf63c60_3.tar.bz2 -C /usr/bin && \
#     sudo rm libstdcxx-ng-7.2.0-hdf63c60_3.tar.bz2

#     # Libtiff 4.0.9
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/libtiff-4.0.9-h28f6b97_0.tar.bz2 && \
#     sudo tar -xjf libtiff-4.0.9-h28f6b97_0.tar.bz2 -C /usr/bin && \
#     sudo rm libtiff-4.0.9-h28f6b97_0.tar.bz2

#     # Libvpx 1.6.1
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/libvpx-1.6.1-h888fd40_0.tar.bz2 && \
#     sudo tar -xjf libvpx-1.6.1-h888fd40_0.tar.bz2 -C /usr/bin && \
#     sudo rm libvpx-1.6.1-h888fd40_0.tar.bz2

#     # Libxcb 1.13
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/libxcb-1.13-h1bed415_1.tar.bz2 && \
#     sudo tar -xjf libxcb-1.13-h1bed415_1.tar.bz2 -C /usr/bin && \
#     sudo rm libxcb-1.13-h1bed415_1.tar.bz2

#     # Libxml2 2.9.8
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/libxml2-2.9.8-hf84eae3_0.tar.bz2 && \
#     sudo tar -xjf libxml2-2.9.8-hf84eae3_0.tar.bz2 -C /usr/bin && \
#     sudo rm libxml2-2.9.8-hf84eae3_0.tar.bz2

#     # Mkl 2018.0.2-1
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/mkl-2018.0.2-1.tar.bz2 && \
#     sudo tar -xjf mkl-2018.0.2-1.tar.bz2 -C /usr/bin && \
#     sudo rm mkl-2018.0.2-1.tar.bz2

#     # Mkl fft 1.0.1
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/mkl_fft-1.0.1-py36h3010b51_0.tar.bz2 && \
#     sudo tar -xjf mkl_fft-1.0.1-py36h3010b51_0.tar.bz2 -C /usr/bin && \
#     sudo rm mkl_fft-1.0.1-py36h3010b51_0.tar.bz2

#     # Mkl random 1.0.1
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/mkl_random-1.0.1-py36h629b387_0.tar.bz2 && \
#     sudo tar -xjf mkl_random-1.0.1-py36h629b387_0.tar.bz2 -C /usr/bin && \
#     sudo rm mkl_random-1.0.1-py36h629b387_0.tar.bz2

#     # Ncurses 6.0
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/ncurses-6.0-h9df7e31_2.tar.bz2 && \
#     sudo tar -xjf ncurses-6.0-h9df7e31_2.tar.bz2 -C /usr/bin && \
#     sudo rm ncurses-6.0-h9df7e31_2.tar.bz2

#     # Openssl 1.0.2
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/openssl-1.0.2o-h20670df_0.tar.bz2 && \
#     sudo tar -xjf openssl-1.0.2o-h20670df_0.tar.bz2 -C /usr/bin && \
#     sudo rm openssl-1.0.2o-h20670df_0.tar.bz2

#     # Pcre 8.42
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/pcre-8.42-h439df22_0.tar.bz2 && \
#     sudo tar -xjf pcre-8.42-h439df22_0.tar.bz2 -C /usr/bin && \
#     sudo rm pcre-8.42-h439df22_0.tar.bz2

#     # Pixman 0.34.0
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/pixman-0.34.0-hceecf20_3.tar.bz2 && \
#     sudo tar -xjf pixman-0.34.0-hceecf20_3.tar.bz2 -C /usr/bin && \
#     sudo rm pixman-0.34.0-hceecf20_3.tar.bz2

#     # Qt 5.9.5
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/qt-5.9.5-h7e424d6_0.tar.bz2 && \
#     sudo tar -xjf qt-5.9.5-h7e424d6_0.tar.bz2 -C /usr/bin && \
#     sudo rm qt-5.9.5-h7e424d6_0.tar.bz2

#     # Readline 7.0
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/readline-7.0-ha6073c6_4.tar.bz2 && \
#     sudo tar -xjf readline-7.0-ha6073c6_4.tar.bz2 -C /usr/bin && \
#     sudo rm readline-7.0-ha6073c6_4.tar.bz2

#     # Sqlite 3.23.1
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/sqlite-3.23.1-he433501_0.tar.bz2 && \
#     sudo tar -xjf sqlite-3.23.1-he433501_0.tar.bz2 -C /usr/bin && \
#     sudo rm sqlite-3.23.1-he433501_0.tar.bz2

#     # Tk 8.6.7
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/tk-8.6.7-hc745277_3.tar.bz2 && \
#     sudo tar -xjf tk-8.6.7-hc745277_3.tar.bz2 -C /usr/bin && \
#     sudo rm tk-8.6.7-hc745277_3.tar.bz2

#     # Xz 5.2.3
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/xz-5.2.3-h5e939de_4.tar.bz2 && \
#     sudo tar -xjf xz-5.2.3-h5e939de_4.tar.bz2 -C /usr/bin && \
#     sudo rm xz-5.2.3-h5e939de_4.tar.bz2

#     # Yaml 0.1.7
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/yaml-0.1.7-had09818_2.tar.bz2 && \
#     sudo tar -xjf yaml-0.1.7-had09818_2.tar.bz2 -C /usr/bin && \
#     sudo rm yaml-0.1.7-had09818_2.tar.bz2

#     # Zlib 1.2.11
# RUN sudo wget https://repo.continuum.io/pkgs/main/linux-64/zlib-1.2.11-ha838bed_2.tar.bz2 && \
#     sudo tar -xjf zlib-1.2.11-ha838bed_2.tar.bz2 -C /usr/bin && \
#     sudo rm zlib-1.2.11-ha838bed_2.tar.bz2

    # Caffe2 Cuda 8.0 Cudnn7 0.8
#RUN sudo wget https://conda.anaconda.org/caffe2/linux-64/caffe2-cuda8.0-cudnn7-0.8.dev-py36_2018.05.14.tar.bz2 && \
#    sudo tar -xjf caffe2-cuda8.0-cudnn7-0.8.dev-py36_2018.05.14.tar.bz2 -C /usr/bin && \
#    sudo rm caffe2-cuda8.0-cudnn7-0.8.dev-py36_2018.05.14.tar.bz2

# Clone CornerNet and CenterNet from GitHub
#RUN pip install --user 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN sudo git clone https://github.com/zhuoyang125/deep-text-recognition-benchmark.git /home/root/deep-text-recognition-benchmark
WORKDIR /home/root/deep-text-recognition-benchmark
RUN mkdir custom_datasets 

# CUDA Setting
ENV FORCE_CUDA="0"

# This will build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
#ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

# # Compiling CornerNet and CenterNet Pooling Layers
# WORKDIR /home/root/CornerNet/models/py_utils/_cpools
# RUN python3 setup.py install --user
# WORKDIR /home/root/CenterNet/models/py_utils/_cpools
# RUN python3 setup.py install --user

# # Compiling NMS for CornerNet and CenterNet
# WORKDIR /home/root/CornerNet/external
# RUN make
# WORKDIR /home/root/CenterNet/external
# RUN make

# # Installing MS COCO APIs for CornerNet and CenterNet
# WORKDIR /home/root/CornerNet/data/coco/PythonAPI
# RUN make
# WORKDIR /home/root/CenterNet/data/coco/PythonAPI
# RUN make
# WORKDIR /home/root

# Set a fixed model cache directory.
## TO DO ##
#WORKDIR /home/appuser

# COPY files from host

#COPY ./coco/annotations /home/root/CornerNet/data/coco/annotations
#COPY ./coco/images /home/root/CornerNet/data/coco/images
#COPY ./coco/annotations /home/root/CenterNet/data/coco/annotations
#COPY ./coco/images /home/root/CenterNet/data/coco/images
#COPY ./cache/nnet/CornerNet /home/root/CornerNet/cache/nnet/CornerNet
#COPY ./cache/nnet/CenterNet-52 /home/root/CenterNet/cache/nnet/CenterNet-52
#COPY ./cache/nnet/CenterNet-104 /home/root/CenterNet/cache/nnet/CenterNet-104
