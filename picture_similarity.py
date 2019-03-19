import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.applications import ResNet50
import cv2
from PIL import Image
import numpy as np
from math import sqrt

model = ResNet50(include_top=False, weights='imagenet', pooling='avg')


def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))


def magnitude(vector):
    return sqrt(dot_product(vector, vector))


def similarity(v1, v2):
    """
    计算余弦相似度
    """
    return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2) + .00000000001)


def template_select(image1, image2):
    image1 = np.expand_dims(np.array(image1.convert('RGB')), axis=0)
    image2 = np.expand_dims(np.array(image2.convert('RGB')), axis=0)
    image1_future = model.predict(image1)[0]
    image2_future = model.predict(image2)[0]
    return similarity(image1_future, image2_future)

