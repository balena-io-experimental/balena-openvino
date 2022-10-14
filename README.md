
 # Balena OpenVino Developer Toolkit

OpenVINO™ is an open-source toolkit for optimizing and deploying AI inference. This project is built around this toolkit, and is aimed to reduce friction in developing AI models on the edge. The aim of this project is to get you started with OpenVINO™ , and allow you to develop, tweak and test models on edge devices. To this end, it includes [Jupyter Notebook](https://jupyter.org/), a popular environment for data science and machine learning applications, and some tools that allow you to interact with the [Intel OpenVino Model Zoo](https://github.com/IntelAI/models).

This is meant to be both a demo of OpenVino running on balena, containing two models, one for [object recognition](https://docs.openvino.ai/latest/omz_models_model_resnet_50_tf.html), and one for [face detection](https://docs.openvino.ai/latest/omz_models_model_face_detection_retail_0044.html), but also a playground where you can [download, convert](https://docs.openvino.ai/latest/omz_tools_downloader.html) and run your models directly on an edge device.  Upon deploying this fleet, in Jupyter, you'll see three notebooks, the two examples, and a template file you can use to work on your model. 

### Hardware Requirements 

* A USB based webcam (or an RTSP video source, more on that later)
* To run this, you'll need an Intel NUC (6th through 11th generation Intel NUC’s are supported, with support for the 12th-gen NUC coming soon), but other 64-bit x86-architecture devices such as laptops, small form factor PC’s, or even x86 single board computers like the UP Board should work too.  Intel NCS2 and Movidius AI accelerators are supported too.

## Architecture 

It's based on Intel's [OpenVino Model Server](https://docs.openvino.ai/latest/ovms_what_is_openvino_model_server.html), a a high-performance system for serving machine learning models. The advantage of using a model server, is that the inference is self contained and independent. 

This enables two scenarios:
* Use as a standalone inference system, on x64 based machines 
* Use as inference server, gathering data/images from multiple lower-power client devices

<img src="https://i.ibb.co/Chd7KM6/structure.png" alt="structure" border="0">

Let's go into a bit of detail with every component of this architecture. 

#### Capture
One of the most popular low-overhead ways to stream images is using [RTSP (Real Time Streaming Protocol)](https://en.wikipedia.org/wiki/Real_Time_Streaming_Protocol).  To keep this as modular as possible, we decided to use this as the default way for image input.

The default way to for image ingest, is connecting a USB camera directly to your NUC and using the [video-capture block]() to stream the images over a (local) RTSP connection.  This project includes a fork of the video-capture block, modified to support x64 devices. 

However, you could also run the capture block on a separate device, let's say a Raspberry Pi with a Pi Camera connected to it or a network cam. Or maybe you want to have a few different image sources you want to run inference on, and use the NUC as a model server itself. As long as they expose an RTSP endpoint any of these sources can be used as image sources for this setup.  While technically possible, don't have any examples for multi-cam and network camera setups yet.

Of course, you can also directly load images or videos directly from the filesystem and run inference on those.

To stream frames from your webcam to an RTSP stream, add this to your `docker-compose.yaml` file: 

```yaml
video-capture:
 build: video-capture 
 network_mode: host 
 privileged: true
 labels:
  io.balena.features.balena-api: '1'
```

#### Model Server
The [OpenVino Model Server (OVMS)](https://docs.openvino.ai/latest/ovms_what_is_openvino_model_server.html)  is the beating heart of this project.  It obviously includes the [OpenVino Inference Engine]() but adds a few more features that make it particularly useful for our use-case.


<img src="https://i.ibb.co/WDBrX5x/oie-transparent.png" alt="oie-transparent" border="0"> 

The model server encapsulates all the packages needed for inference and model execution and **Device Plugins** to talk to the device you want to run inference on (can be CPU, GPU or AI Accelerator), and exposes either a gRPC or REST endpoint. This means you simply need to feed it the model input, in this case images, but could be anything your model is trained to accept, and wait for an inference result. The scheduling is done automatically, so it's able to accept and  respond to requests to multiple models at the same time.  

The **Configuration Monitoring** and **Model Management** parts allow us to dynamically load new models and change the configuration file on the fly. 

Both of these features, running multiple models at the same time, and the dynamic loading of models enable very powerful features for embedded and edge devices. Think of a robot that might need to do segmentation at some point, and then pull up an object detection model in another context. 
 

The configuration format for the OpenVino Model server looks like this. You can find more information and details in the [official Intel documentation.](https://docs.openvino.ai/latest/ovms_docs_multiple_models.html)


```json
{
  "model_config_list": [
    {
      "config": {
        "name": "face-detection",
        "base_path": "/usr/openvino/model/face-detection"
      }
    },
    {
      "config": {
        "name": "resnet",
        "base_path": "/usr/openvino/model/resnet"
      }
    }
  ]
}
```

For a model to be considered and loaded by the OpenVino model server, it needs to respect the following file-structure:
```
base_path
└── face-detection <-- model name
    └── 1 <-- version number
        ├── face-detection-retail-0004.bin <-- OpenVino IR model binary 
        └── face-detection-retail-0004.xml <-- Model configuration
```
To add the OpenVino model server to your fleet you simply need to add this to your `docker-compose.yaml` file:
```yaml
model-server:
  image: openvino/model_server:latest
  command: /ovms/bin/ovms --config_path /usr/openvino/code/config.json --port 9000
  network_mode: host
  ports:
    - 9000:9000
  volumes:
    - models:/usr/openvino

```


#### Shared Volume 
As we've seen, OVMS needs a couple of assets to be able to run inference, one is the `config.json` file, and the other are the model binaries and configuration files. We have already added those for two models, you can check that out to see how the filesystem hierarchy is structured.

Adding these files on a shared volume allows you to edit the OpenVino model server configuration files, download models, and convert them directly from the Jupyter. This means you can change things on the fly,  and since OVMS supports hot-reload, these changes will be immediately reflected in the model server,  allowing you to instantaneously run inference on your newly downloaded model. 

#### Jupyter
Jupyter is an interactive web based development environment for Python. It is very popular in data science applications. 


## How to use
This repository contains everything you need to get started with OpenVino on the edge. The `docker-compose.yaml`file comes pre-configured with, OVMS, a local RTSP streamer for USB webcams, Jupyter Notebook, and two example models, one for object detection and one for face recognition. 

[![balena deploy button](https://www.balena.io/deploy.svg)](https://dashboard.balena-cloud.com/deploy?repoUrl=https://github.com/cristidragomir97/openvino-balena/)

Before you start, make sure a webcam is connected to your device. Deploy this to a fleet, and navigate to http://<YOUR_DEVICE_IP>:8888. You should be greeted by the Jupyter environment.

Once logged in into Jupyter, you'll be ready to run the included demos, or add your own. You can see the file tree on the sidebar on the left.

![](https://i.ibb.co/sCzhFKS/Screenshot-2022-09-20-at-18-42-27.png)

To run the cells in the notebook you can either use the notebook's toolbar:

![](https://i.ibb.co/C8C4xyX/toolbar.png)

Or go to the "Kernel" tab in the menubar, and select "Restart Kernel and Run all Cells".

![](https://i.ibb.co/25z1Dry/Screenshot-2022-09-20-at-18-42-32.png)

Now scroll down to the end of the page and you should see a live camera feed with your model running. 


## How to import pre-trained modelsfrom OpenVino Model Zoo
1. Browse [OpenVino Model Zoo](https://docs.openvino.ai/latest/omz_models_group_public.html). Let's say we want to try and use `yolo-v3-tf` in our solution.
2. Open an a terminal window inside Jupyter by opening the command palette (Command/Ctrl + Shift + C) and selecting `New Terminal` and navigate to ` /usr/openvino/model/`
3. Use `omz_downloader` to download and convert the model. 
* `omz_downloader --name yolo-v3-tf`
* `omz_converter --name yolo-v3-tf`
4. In order to use a model, you'll need three files, the model binary (.bin), the model manifest (.xml) and the layer mapping file (.mapping). Move the files that `omz_converter` created to the correct folder `/usr/openvino/model/<MODEL_NAME>/<MODEL_VERSION>`
* `mv /usr/openvino/model/public/yolo-v3-tf/FP32/yolo-v3-tf.xml /usr/openvino/model/yolo-v3-tf/1/yolo-v3-tf.xml` 
* `mv /usr/openvino/model/public/yolo-v3-tf/FP32/yolo-v3-tf.bin /usr/openvino/model/yolo-v3-tf/1/yolo-v3-tf.bin` 
* `mv /usr/openvino/model/public/yolo-v3-tf/FP32/yolo-v3-tf.mapping /usr/openvino/model/yolo-v3-tf/1/yolo-v3-tf.mapping` 
5. Add the model to the `config.json` file
```json
 {
         "config":{
            "name":"yolo-v3-tf",
            "base_path":"/usr/openvino/model/yolo-v3-tf"
         }
      },
```
6. The OpenVino model server service will immediatly see the changes and begin serving your model. 

7. Check the `template.ipynb`file for an example of how to do inference on your newly imported model





