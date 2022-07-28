

import cv2, io, time, imutils, datetime
import traceback
from imutils.video import VideoStream
import numpy as np
from classes import imagenet_classes
from ovmsclient import make_grpc_client

#import grpc
#from tensorflow import make_tensor_proto, make_ndarray
#from tensorflow_serving.apis import predict_pb2
#from tensorflow_serving.apis import prediction_service_pb2_grpc

GRPC_URL = "localhost:9000"
RTSP_URL = "rtsp://localhost:8554/server"
BATCH_SIZE = 1
MODEL_NAME = 'face-detection'
WIDTH = 640
HEIGHT = 480
FPS = 30


def crop_resize(img,cropx,cropy):
    y,x,c = img.shape
    if y < cropy:
        img = cv2.resize(img, (x, cropy))
        y = cropy
    if x < cropx:
        img = cv2.resize(img, (cropx,y))
        x = cropx
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx,:]

def __draw_label(img, text, pos, bg_color):
   font_face = cv2.FONT_HERSHEY_SIMPLEX
   scale = 0.4
   color = (0, 0, 0)
   thickness = cv2.FILLED
   margin = 2
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)

   end_x = pos[0] + txt_size[0][0] + margin
   end_y = pos[1] - txt_size[0][1] - margin

   cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
   cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

def prep_image(img, size = 224):
    img = cv2.resize(img, (WIDTH, HEIGHT))
    img = img.astype('float32')
    img = img[:, :, [2, 1, 0]]
    img = img.transpose(2,0,1).reshape(1,3, HEIGHT, WIDTH)
    #img = crop_resize(img, size, size)
    #img = img.transpose(2,0,1).reshape(1,3,size,size)
    return img

def do_face_detection(stub, img):
   # output = client.predict({"0": img}, "resnet")
    #result_index = np.argmax(output[0])
    #predicted_class = imagenet_classes[result_index]
    #return predicted_class
    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME
    print("\nRequest shape", img.shape)
    request.inputs["0"].CopyFrom(make_tensor_proto(img, shape=(img.shape)))
    start_time = datetime.datetime.now()
    result = stub.Predict(request, 10.0) # result includes a dictionary with all model outputs
    end_time = datetime.datetime.now()

    for thing in result.outputs: print(thing)
    output = make_ndarray(result.outputs[86])
    print("Response shape", output.shape)

    for y in range(0,img.shape[0]):  # iterate over responses from all images in the batch
        img_out = img[y,:,:,:]

        print("image in batch item",y, ", output shape",img_out.shape)
        img_out = img_out.transpose(1,2,0)
        for i in range(0, 200*BATCH_SIZE-1):  # there is returned 200 detections for each image in the batch
            detection = output[:,:,i,:]
            # each detection has shape 1,1,7 where last dimension represent:
            # image_id - ID of the image in the batch
            # label - predicted class ID
            # conf - confidence for the predicted class
            # (x_min, y_min) - coordinates of the top left bounding box corner
            #(x_max, y_max) - coordinates of the bottom right bounding box corner.
            if detection[0,0,2] > 0.5 and int(detection[0,0,0]) == y:  # ignore detections for image_id != y and confidence <0.5
                print("detection", i , detection)
                x_min = int(detection[0,0,3] * WIDTH)
                y_min = int(detection[0,0,4] * HEIGHT)
                x_max = int(detection[0,0,5] * WIDTH)
                y_max = int(detection[0,0,6] * HEIGHT)
                # box coordinates are proportional to the image size
                print("x_min", x_min)
                print("y_min", y_min)
                print("x_max", x_max)
                print("y_max", y_max)

                img_out = cv2.rectangle(cv2.UMat(img_out), tuple(x_min,y_min), tuple(x_max,y_max),(0,0,255),1)
    
    return img_out

def resnet(client, img):
    output = client.predict({"0": img}, "resnet")
    result_index = np.argmax(output[0])
    predicted_class = imagenet_classes[result_index]
    return predicted_class

def publish_result_stream(img):
    return None


if __name__ == "__main__":
    grpc_client = make_grpc_client(GRPC_URL)
    rtsp_stream = VideoStream(RTSP_URL).start()
    print("* RTSP Stream succesfully opened")
    print("* gRPC Socket succesfully opened")

    while True:
        try: 
            frame = rtsp_stream.read()
            try: 
                if frame is not None:
                    aux_frame = prep_image(frame)
                    result = resnet(grpc_client, aux_frame)
                    __draw_label(frame, str(result), (20,20), (255,0,0))
                    print(result)
                    out.write(frame)

                else: 
                   continue 
                
                time.sleep(1/30)
            except Exception as e:
                print("error while running inference", e)
                print(traceback.format_exc())
          
        except Exception as e:
            print("error while grabbing RTSP stream", e)
