FROM balenalib/intel-nuc-ubuntu:focal

RUN install_packages wget libffi-dev gfortran git cmake build-essential \
    python3 python3-pip python3-dev python3-dev python3-numpy  \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-ugly gstreamer1.0-rtsp  

WORKDIR /usr/openvino
##RUN  wget -P model/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin
##RUN  wget -P model/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml

WORKDIR /usr/openvino
RUN mkdir -p /usr/openvino/code/notebooks
WORKDIR /usr/openvino/code/notebooks
RUN wget https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/python/classes.py
COPY ./config.json /usr/openvino/code/config.json

COPY ./notebooks/face_detection.ipynb /usr/openvino/code/notebooks/face_detection.ipynb
COPY ./notebooks/resnet.ipynb /usr/openvino/code/notebooks/resnet.ipynb
COPY ./notebooks/yolov5.ipynb /usr/openvino/code/notebooks/yolov5.ipynb


RUN pip3 install --upgrade pip
RUN pip3 install ovmsclient imutils setuptools cffi jupyterlab matplotlib pandas opencv-python ipywidgets opencv_jupyter_ui openvino-dev tensorflow-serving-api

WORKDIR /usr/openvino/model

#RUN omz_downloader --name yolo-v3-tf \
#    && omz_converter --name yolo-v3-tf \
#    && mkdir -p 1 \
#    && mv /usr/openvino/model/public/yolo-v3-tf/FP32/yolo-v3-tf.xml /usr/openvino/model/1/yolo-v3-tf.xml \
#    && mv /usr/openvino/model/public/yolo-v3-tf/FP32/yolo-v3-tf.bin /usr/openvino/model/1/yolo-v3-tf.bin \
#    && mv /usr/openvino/model/public/yolo-v3-tf/FP32/yolo-v3-tf.mapping /usr/openvino/model/1/yolo-v3-tf.mapping \
#    && rm -rf /usr/openvino/model/public

RUN  wget -P resnet/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin
RUN  wget -P resnet/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml

RUN  wget -P face-detection/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.bin
RUN  wget -P face-detection/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.xml

COPY  ./yolov5n/yolov5n.bin  /usr/openvino/model/yolov5n/1/yolov5n.bin
COPY  ./yolov5n/yolov5n.mapping /usr/openvino/model/yolov5n/1/yolov5n.mapping
COPY  ./yolov5n/yolov5n.xml /usr/openvino/model/yolov5n/1/yolov5n.xml 
COPY  ./yolov5n/yolov5n.yaml  /usr/openvino/model/yolov5n/1/yolov5n.yaml

CMD jupyter lab /usr/openvino/ --allow-root --ip 0.0.0.0
