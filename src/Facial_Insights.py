import argparse
import os
import numpy as np
import sys
import json
from os import listdir
import csv
from os.path import isfile, join

from face_network import create_face_network
import cv2
import argparse
from keras.optimizers import Adam, SGD


from keras.models import load_model

from emotion_model import *
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input
from tempfile import TemporaryFile
from keras.backend import tf as ktf

from pprint import pprint


import urllib.request
import shutil
import h5py

import dlib
import matplotlib.pyplot as plt
import tensorflow as tf
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

import inception_resnet_v1

from tqdm import tqdm
from pathlib import Path
from keras.utils.data_utils import get_file

import sys

from keras.models import load_model
from keras.models import Sequential
from keras.models import load_model
import numpy as np
from time import gmtime, strftime


os.environ['CUDA_VISIBLE_DEVICES'] = ''

class Person_Input():
    global model_creation_path
    global shape_detector


    global detection_model_path
    global emotion_model_path
    global gender_model_path
    global age_model_path
    global ethnic_model_path
    global ETHNIC

    model_creation_path = "./models/"
    shape_detector = "shape_predictor_68_face_landmarks.dat"


    detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
    #For another way to calculate age. 
    #age_model_path = '../trained_models/age_models/weights.25000-0.03.hdf5'
    ethnic_model_path = '../trained_models/ethnic_models/weights_ethnic.hdf5'

    ETHNIC = {0: 'Asian', 1: 'Caucasian', 2: "African", 3: "Hispanic"}



    def __init__(self, path_to_image):
        self.path_to_image = path_to_image

    def get_emotion(self, image_path_, face_detection, emotion_classifier, gender_classifier):
        
        emotion_labels = get_labels('fer2013')

        # hyper-parameters for bounding boxes shape
        # change for variance
        emotion_offsets = (20, 40)
        emotion_offsets = (0, 0)

        # loading models
        # getting input model shapes for inference
        emotion_target_size = emotion_classifier.input_shape[1:3]
        
        # loading images
        rgb_image = load_image(image_path_, grayscale=False)
        gray_image = load_image(image_path_, grayscale=True)
        gray_image = np.squeeze(gray_image)
        gray_image = gray_image.astype('uint8')

        faces = detect_faces(face_detection, gray_image)
        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            rgb_face = rgb_image[y1:y2, x1:x2]

            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
                rgb_face = rgb_image[y1:y2, x1:x2]
            except:
                continue

            rgb_face = preprocess_input(rgb_face, False)
            rgb_face = np.expand_dims(rgb_face, 0)

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
            emotion_text = emotion_labels[emotion_label_arg]

            return emotion_text

    def get_gender(self, image_path_, face_detection, emotion_classifier, gender_classifier):
       
        gender_labels = get_labels('imdb')
    
        # hyper-parameters for bounding boxes shape
        # change for variance
        gender_offsets = (30, 60)
        gender_offsets = (10, 10)
        emotion_offsets = (20, 40)
        emotion_offsets = (0, 0)

        # loading models
        # getting input model shapes for inference
        gender_target_size = gender_classifier.input_shape[1:3]

        # loading images
        rgb_image = load_image(image_path_, grayscale=False)
        gray_image = load_image(image_path_, grayscale=True)
        gray_image = np.squeeze(gray_image)
        gray_image = gray_image.astype('uint8')

        faces = detect_faces(face_detection, gray_image)
        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            rgb_face = rgb_image[y1:y2, x1:x2]

            try:
                rgb_face = cv2.resize(rgb_face, (gender_target_size))
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            rgb_face = preprocess_input(rgb_face, False)
            rgb_face = np.expand_dims(rgb_face, 0)
            gender_prediction = gender_classifier.predict(rgb_face)
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = gender_labels[gender_label_arg]

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            
   
            return gender_text

    #another way to find age. 
    def get_age2(self, sess,age,gender, train_mode,images_pl, image_path):

        #image_path = '/Users/adelwang/Documents/Hackery/Gender-Age-Expression/GenderExpression2/images/will.jpg'
        # for face detection
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        fa = FaceAligner(predictor, desiredFaceWidth=160)

        # load model and weights
        img_size = 160

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - 0.4 * w), 0)
            yw1 = max(int(y1 - 0.4 * h), 0)
            xw2 = min(int(x2 + 0.4 * w), img_w - 1)
            yw2 = min(int(y2 + 0.4 * h), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
            faces[i, :, :, :] = fa.align(input_img, gray, detected[i])
            # faces[i,:,:,:] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

        if len(detected) > 0:
            # predict ages and genders of the detected faces
            ages, genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})
            #print(int(ages[i]))
            #print("F" if genders[i] == 0 else "M")
        return int(ages[i])


    def get_age(self, aligned_images, model_path):
        with tf.Graph().as_default():
            sess = tf.Session()
            images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
            images = tf.map_fn(lambda frame: tf.reverse_v2(frame, [-1]), images_pl) #BGR TO RGB
            images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images)
            train_mode = tf.placeholder(tf.bool)

            age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                         phase_train=train_mode,
                                                                         weight_decay=1e-5)
       
          
            age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
            age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_))
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)
            saver = tf.train.Saver()
            
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("restore and continue training!")
            else:
                pass
            

            return sess.run(age, feed_dict={images_pl: aligned_images, train_mode: False})
            #return sess.run([age, gender], feed_dict={images_pl: aligned_images, train_mode: False})

    def predict_ethnic(self, name):
        means = np.load('means_ethnic.npy')

        model = create_face_network(nb_class=4, hidden_dim=512, shape=(224, 224, 3))
        model.load_weights('weights_ethnic.hdf5')

        im = cv2.imread(name, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        im = cv2.resize(im, (224, 224))
        im = np.float64(im)
        im /= 255.0
        im = im - means
        #return model.predict(np.array([im]))
        return model.predict(np.array([im]))

    def load_image(self, image_path, shape_predictor):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_predictor)
        fa = FaceAligner(predictor, desiredFaceWidth=160)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # image = imutils.resize(image, width=256)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 2)
        rect_nums = len(rects)
        XY, aligned_images = [], []
        if rect_nums == 0:
            aligned_images.append(image)
            return aligned_images, image, rect_nums, XY
        else:
            for i in range(rect_nums):
                aligned_image = fa.align(image, gray, rects[i])
                aligned_images.append(aligned_image)
                (x, y, w, h) = rect_to_bb(rects[i])
                image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
                XY.append((x, y))
            return np.array(aligned_images), image, rect_nums, XY

    def load_network(self, model_path):
        sess = tf.Session()
        images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
        images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_pl)
        train_mode = tf.placeholder(tf.bool)
        age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                     phase_train=train_mode,
                                                                     weight_decay=1e-5)
        gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            pass
        return sess,age,gender, train_mode,images_pl

    def face_detection(self, detection_model_path):
        global face_detection
        face_detection = load_detection_model(detection_model_path)
        return face_detection


def getFacialInsights(self, path_to_file):

    if path_to_file is None:
        return 0

    image_path = '/Users/adelwang/Documents/Hackery/Gender-Age-Expression/GenderExpression2/images/obama1.jpg'
    path_to_file = '/Users/adelwang/Documents/Hackery/Gender-Age-Expression/GenderExpression2/images'

    shape_detector = "shape_predictor_68_face_landmarks.dat"
    model_path = "./models/"

    person = Person_Input(path_to_file)

    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    gender_classifier = load_model(gender_model_path, compile=False)

    #aligned_image, image, rect_nums, XY = person.load_image(image_path, shape_detector)

    five_insights = [None]*5
    count = 0



    sess, age_, gender_, train_mode,images_pl = person.load_network("./models")

    for f in listdir(path_to_file):
        if isfile(join(path_to_file, f)) and not f.startswith('.'):
            image_path_= join(path_to_file, f)
            emotion = person.get_emotion(image_path_, face_detection, emotion_classifier, gender_classifier)
            gender = person.get_gender(image_path_, face_detection, emotion_classifier, gender_classifier)
            #Another way to get age. 
            #sess, age, gender_, train_mode,images_pl = person.load_network("./models")
            #aligned_image, image, rect_nums, XY = person.load_image(image_path_, shape_detector)
            #age = person.get_age(aligned_image, shape_detector)
            age = person.get_age2(sess,age_,gender_,train_mode,images_pl, image_path_)
            result = person.predict_ethnic(image_path)
            ethnicity = ETHNIC[np.argmax(result)]
            #one_person_insight_ = {'age':int(age), 'gender':gender, 'expression':emotion, 'ethnicity': ethnicity}
            one_person_insight_ = {'age':int(age), 'gender':gender, 'expression':emotion}
            one_person_insight = json.dumps(one_person_insight_)
            five_insights[count] = one_person_insight
            count += 1
    #print(five_insights)
    return five_insights
    

#pass in the file name
getFacialInsights("fileName", "fileName")


