import sys

import cv2
from keras.models import load_model
import numpy as np
from time import gmtime, strftime
import csv
import os


from keras import backend as K
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input


def emotion_model(self):
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    # Expect an image through terminal 
    # parameters for loading data and images
    image_path = sys.argv[1]
    detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
    emotion_labels = get_labels('fer2013')
    gender_labels = get_labels('imdb')
    font = cv2.FONT_HERSHEY_SIMPLEX

    # hyper-parameters for bounding boxes shape
    # change for variance
    gender_offsets = (30, 60)
    gender_offsets = (10, 10)
    emotion_offsets = (20, 40)
    emotion_offsets = (0, 0)


    # loading models
    face_detection = load_detection_model(detection_model_path)
    print("444444")
    emotion_classifier = load_model(emotion_model_path, compile=False)
    print(emotion_classifier)
    print("55555")
    gender_classifier = load_model(gender_model_path, compile=False)

    return emotion_classifier
#K.clear_session()
