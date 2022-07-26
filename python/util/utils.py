from __future__ import division
import math as mt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

def crop_frame(frame, x, y, w, h):
    """
    Crop the frame
    """
    return frame[y:y+h, x:x+w]


def get_main__label_detection(detections, classes):
    """
    Get the main object detection
    """

    mainDetection = []
    for detection in detections:
        if detection is not None and detection.get('probability') > 0.3: #and detection.get('tagName') in clasesCopy 
            mainDetection.append(detection)
            # clasesCopy.remove(detection.get('tagName'))
    return mainDetection
