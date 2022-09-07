

# Balena OpenVino Developer Toolkit

OpenVINO™ is an open-source toolkit for optimizing and deploying AI inference. This project is built around this toolkit, and is aimed to reduce friction in developing AI models on the edge. The aim of this project is to get you started with OpenVINO™ , and allow you to develop, tweak and test models on edge devices. To this end, it includes Jupyter Notebook, a popular environment for data science and machine learning applications, and some tools that allow you to interact with the [Intel OpenVino Model Zoo]().

The way this is built allows you to enable two scenarios:
* 
* 

Upon deploying this fleet, in jupyter, you'll see three notebooks, two examples, and a template file you can use to work on your model. 

### Hardware Requirements 

* A USB based webcam (or an RTSP video source, more on that later)
* To run this, you'll need an Intel NUC (6th through 11th generation Intel NUC’s are supported, with support for the 12th-gen NUC coming soon), but other 64-bit x86-architecture devices such as laptops, small form factor PC’s, or even x86 single board computers like the UP Board should work too.  Intel NCS2 and Movidius AI accelerators are supported too.

## Architecture 

It's based on Intel's [OpenVino Model Server](https://docs.openvino.ai/latest/ovms_what_is_openvino_model_server.html), a a high-performance system for serving machine learning models. The advantage of using a model server, is that the inference is self contained and independent. This leads to a couple of advantages we are going to talk about later. 

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
	"model_config_list":[
		{
			"config":{
				"name":"face-detection",
				"base_path":"/usr/openvino/model/face-detection"
			}
		},
		{
			"config":{
				"name":"resnet",
				"base_path":"/usr/openvino/model/resnet"
			}
		}
	]
}
```

For a model to be considered and loaded by the OpenVino model server, it needs to respect the following file-structure. 
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


## How to use
This repository contains everything you need to get started with OpenVino on the edge. The `docker-compose.yaml`file comes pre-configured with, OVMS, a local RTSP streamer for USB webcams, Jupyter Notebook, and two example models, one for object detection and one for face recognition. 

Deploy this to a fleet, connect a USB webcam to your NUC, and look for the Jupyter Notebook URL on the dashboard.  Once logged in into Jupyter, you'll be ready to run the included demos, or add your own. 

## How to extend
You'll probably get bored of the two included examples quite quickly. 
