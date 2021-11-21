import os
from django.apps import AppConfig
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np

class Covid19Config(AppConfig):
    name = 'covid19'

class ResNetModelConfig(AppConfig):
    name = 'resnetAPI'
    MODEL_FILE = os.path.join(settings.MODELS, "resnet_chest.h5")
    model = keras.models.load_model(MODEL_FILE)

class VGGModelConfig(AppConfig):
    name = 'vggAPI'
    MODEL_FILE = os.path.join(settings.MODELS, "vgg_chest.h5")
    model = keras.models.load_model(MODEL_FILE)

class InceptionModelConfig(AppConfig):
    name = 'inceptionv3_chestAPI'
    MODEL_FILE = os.path.join(settings.MODELS, "inceptionv3_chest.h5")    
    model = keras.models.load_model(MODEL_FILE)

class ExceptionModelConfig(AppConfig):
    name = 'xception_chestAPI'
    MODEL_FILE = os.path.join(settings.MODELS, "xception_chest.h5")    
    model = keras.models.load_model(MODEL_FILE)