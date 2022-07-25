from __future__ import division
from models import *
from predict import predict_image
import os
import sys
import argparse
import cv2
from PIL import Image
import torch
from torch.autograd import Variable
import predict
import numpy as np
from util.utils import  *

def Convertir_RGB(img):
    # Convertir Blue, green, red a Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def Convertir_BGR(img):
    # Convertir red, blue, green a Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="../labels.txt", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--webcam", type=int, default=1,  help="Is the video processed video? 1 = Yes, 0 == no" )
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--directorio_video", type=str, help="Directorio al video")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args() #load (arg)
    
    isThereGraphicCard = torch.cuda.is_available()

    device = torch.device("cuda" if isThereGraphicCard else "cpu")

    # Crear a video without labels
    out = cv2.VideoWriter('./videos/SinRecuadro.mp4', 0, 60.0, (800, 620))

    # define a video capture object
    vid = cv2.VideoCapture(0)
    # Define the label classes detections
    classes = load_classes(opt.class_path)
    #Assign a color for each classes
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    #load Tensor
    Tensor = torch.cuda.FloatTensor if isThereGraphicCard else torch.FloatTensor

    while(True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        if ret is False:  # If the fram is avaible
            break
    
        # Display the resulting frame
        cv2.imshow('frameSon', frame)
        #Save frame without information
        out.write(frame)

        # RGBimg = Convertir_RGB(frame)
        # Convert to image to process in the model
        imgTensor = Image.fromarray(frame)
        detections = predict.predict_image(imgTensor)
        detections  = get_main__label_detection(detections, classes)
        for detection in detections:
            if detection is not None:
                print(detection)
                boundig_boxes = detection.get('boundingBox')
                x1, y1, w, h = boundig_boxes.values()
                
                y1 *= imgTensor.height
                x1 *= imgTensor.width
                w *= imgTensor.width
                h *= imgTensor.height

                x2 = x1 + w
                y2 = y1 + h                

                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), [0, 255, 0], 5)
                cv2.putText(frame, detection.get('tagName'), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, [228, 54, 16], 5) #Nombre de la clase
                cv2.imshow('frame', frame)


        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    out.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
   