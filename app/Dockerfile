FROM balenalib/intel-nuc-ubuntu:focal

RUN install_packages wget python3 python3-pip python3-dev gfortran git cmake build-essential libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-ugly gstreamer1.0-rtsp python3-dev python3-numpy libffi-dev

#WORKDIR /usr/opencv
#RUN git clone --depth=1 -b 4.5.4 https://github.com/opencv/opencv \
#    && cd opencv \
#    && mkdir build && cd build \
#    && cmake -D CMAKE_INSTALL_PREFIX=/usr -D WITH_GSTREAMER=ON .. \
#    && make -j8 \
#    && make install 

WORKDIR /usr/models
RUN  wget -P model/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin
RUN  wget -P model/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml
COPY config.json /usr/models/config.json

WORKDIR /usr/src/app/
RUN wget https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/python/classes.py

RUN pip3 install ovmsclient imutils setuptools cffi jupyterlab pi-ina219 matplotlib pandas opencv-python

COPY main.py /usr/src/app/main.py

CMD jupyter lab /usr --allow-root --ip 0.0.0.0




