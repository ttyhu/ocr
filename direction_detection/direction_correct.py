import os
import pickle
import time

import cv2 as cv
import numpy as np
import math
from math import fabs, sin, radians, cos

from PIL import Image
from keras import regularizers, Model
from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D, Dense
from pdf2image import convert_from_path


def cr():
    base_model = VGG16(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    x = Dense(128, kernel_regularizer=regularizers.l2(0.000), activation='relu')(x)
    predictions = Dense(2, activation='sigmoid')(x)
    model = Model(input=base_model.input, output=predictions)
    return model


def rotate_img(image, degree):
    degree = -degree
    img1 = np.array(image.convert('RGB'))
    height, width = img1.shape[:2]

    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))  # 这个公式参考之前内容
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
    matRotation[1, 2] += (heightNew - height) / 2

    imgRotation = cv.warpAffine(img1, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return Image.fromarray(imgRotation)


def fourier_demo(image1, ft):
    a_img = image1.copy()
    image1.thumbnail((1000, 1000), Image.ANTIALIAS)
    ft_list = ['FT007006003001', 'FT007006005001', 'FT007006007001', 'FT007006008001', 'FT007006009001',
               'FT007006010001', 'FT007006016001', 'FT007006022001', 'FT007008002001']
    if ft not in ft_list:
        angle_2 = rotate_classification(image1, 0)
        # print('angle:', angle_2)
        rotated = rotate_img(a_img, angle_2)
        return rotated
    else:
        # 1、读取文件，灰度化
        # image1 = Image.open(image_path)
        img = np.array(image1.convert('RGB'))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 2、图像延扩
        h, w = img.shape[:2]
        new_h = cv.getOptimalDFTSize(h)
        new_w = cv.getOptimalDFTSize(w)
        right = new_w - w
        bottom = new_h - h
        nimg = cv.copyMakeBorder(gray, 0, bottom, 0, right, borderType=cv.BORDER_CONSTANT, value=0)

        # 3、执行傅里叶变换，并过得频域图像
        f = np.fft.fft2(nimg)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift))

        # 二值化
        magnitude_uint = magnitude.astype(np.uint8)
        ret, thresh = cv.threshold(magnitude_uint, 11, 255, cv.THRESH_BINARY)

        # 霍夫直线变换
        lines = cv.HoughLinesP(thresh, 2, np.pi/180, 30, minLineLength=40, maxLineGap=100)

        # # 创建一个新图像，标注直线
        # lineimg = np.ones(nimg.shape, dtype=np.uint8)
        # lineimg = lineimg * 255

        # new = np.ones(nimg.shape, dtype=np.uint8)
        # new = new * 255

        max_len = 0
        index = 0
        if len(lines) > 0:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                dis = (x1-x2)**2 + (y1-y2)**2
                if x2 - x1 != 0 and y2 - y1 != 0:
                    if dis > max_len:
                        max_len = dis
                        index = i
            #         cv.line(new, (x1, y1), (x2, y2), (0, 255, 0), 2)

            x1, y1, x2, y2 = lines[index][0]

            # # show霍夫直線
            # cv.line(lineimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Image.fromarray(lineimg).show()

            theta = (x2 - x1) / (y2 - y1)
            angle_1 = math.atan(theta)
            angle_1 = angle_1 * (180 / math.pi)/(w/h)
            if abs(1/theta) < 0.1:
                angle_1 = 0
        else:
            angle_1 = 0
        angle_2 = rotate_classification(image1, angle_1)
        # print('angle:', angle_1+angle_2)
        rotated = rotate_img(a_img, angle_1+angle_2)
    return rotated


def rotate_classification(image, angle):
    # label_dict = pickle.load(open('model/label_rev_dict.pkl', 'rb'))
    model = cr()
    model.load_weights('direction_detection/model/weights-01-0.9872881355932204.hdf5')
    img1 = image.resize((224, 224)).convert('RGB')
    img1 = rotate_img(img1, angle)
    img1 = np.array(img1) / 255
    img1 = np.expand_dims(img1, axis=0)
    label = np.argmax(model.predict(img1)[0])
    model_1 = cr()
    if label == 1:
        label_dict_1 = pickle.load(open('direction_detection/model/90/label_rev_dict.pkl', 'rb'))
        model_1.load_weights('direction_detection/model/90/{}'.format(os.listdir('direction_detection/model/90')[1]))
        label_1 = np.argmax(model_1.predict(img1)[0])
        label_1 = label_dict_1[label_1]
    else:
        label_1 = 0
    # img1 = rotate_img(image, -int(label_1))
    # print('1111111111111111111')
    return -int(label_1)    # img1


# if __name__ == '__main__':
#     for file_name in os.listdir('111'):
#         try:
#             if not os.path.exists('rotated_images/' + os.path.splitext(file_name)[0] + '_2.jpg'):
#                 # print(file_name)
#                 start = time.time()
#                 path = '111/' + file_name
#                 if file_name.lower().endswith('pdf'):
#                     old = convert_from_path(path)[0]
#                 else:
#                     old = Image.open(path)
#                 new = fourier_demo(old, ft='FT00700600300')
#                 end = time.time()
#                 # print(end-start)
#                 new.save('rotated_images/' + os.path.splitext(file_name)[0] + '_2.jpg')
#         except Exception as ex:
#             print(ex)
