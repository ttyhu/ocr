import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from picture_similarity import template_select
from utils.convert_number import money_transform
from utils.convert_time import convert_time

from template_match import match_template, affine_transform, match_template_calibration_test1
from get_text_area import get_area
from ctpn.ctpn_blstm_test import text_predict
from densent_ocr.densenet_ocr_test import predict
from img_cls.predict_2 import predict_ft
from select_infomodel import select
from list_link_same_line import link_same_line, get_big_text

import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import pickle
from sanic import Sanic, response

app = Sanic(__name__)

info = pickle.load(open('info.pkl', 'rb'))

large_FT = ['FT001001001002', 'FT001001001003', 'FT001001001005', 'FT001003002001', 'FT001001002001', 'FT001001003001']
nlp_FT = ['FT007008002001', 'FT007008001001', 'FT007006022001', 'FT007006010001', 'FT007006009001', 'FT007006008001',
          'FT007006007001', 'FT007006006001', 'FT007006005001', 'FT007006004001', 'FT007006003001', 'FT007005001001',
          'FT007004002001', 'FT007003002002', 'FT007002002001', 'FT007001002001', 'FT001002003001', 'FT001001001001',
          'FT001001001002', 'FT001001001003', 'FT001001001004', 'FT001001001005', 'FT001001001006', 'FT001001001007']


@app.route('/fce', methods=['POST'])
def get_text(request):
    img_path = request.form.get('img_path')
    FT = request.form.get('FT')
    task_id = request.form.get('task_id')
    page = request.form.get('page')
    try:
        if img_path.lower().endswith('.pdf'):
            input_img = convert_from_path(img_path)[int(page) - 1]
        else:
            input_img = Image.open(img_path)
        scale = 0.3
        pts1 = []
        pts2 = []
        print(11111111111111111)
        FT = predict_ft(input_img, FT)
        if FT in nlp_FT:
            f = select(FT)
            texts = f.extract_info(img_path, page, FT)
            blank = 0
            for i in texts:
                if i == '':
                    blank += 1
            if blank == len(texts) - 1:
                return response.json(
                    {'result': 'false', 'message': '请求失败', 'taskid': task_id, 'fields': texts, 'FT': FT})
            else:
                print(3333333333333333333)
                return response.json(
                    {'result': 'true', 'message': '请求成功', 'taskid': task_id, 'fields': texts, 'FT': FT})
        templates = os.listdir('template_images/{}'.format(FT))
        if len(templates) == 1:
            template = templates[0]
        else:
            template = ''
            similarity = float('-inf')
            for i in templates:
                template_similarity = template_select(input_img.resize((224, 224)),
                                                      Image.open(
                                                          'template_images/{}/{}/template.jpg'.format(FT, i)).resize(
                                                          (224, 224)))
                print(template_similarity)
                if template_similarity > similarity:
                    similarity = template_similarity
                    template = i
        print(template)
        for index, i in enumerate(sorted(os.listdir('template_images/{}/{}/templates'.format(FT, template)))):
            pts = match_template('template_images/{}/{}/templates/'.format(FT, template) + i, input_img)
            if pts1 == [] and len(pts[0]) != 0:
                pts1 = pts[0]
                pts2 = pts[1]
            elif len(pts[0]) != 0:
                pts1 = np.append(pts1, pts[0], axis=0)
                pts2 = np.append(pts2, pts[1], axis=0)

        img = affine_transform(input_img.convert('RGB'), 'template_images/{}/{}/template.jpg'.format(FT, template),
                               pts1=pts1,
                               pts2=pts2)
        dynamic_picture_1 = os.listdir('template_images/{}/{}/calibration/'.format(FT, template))[0]
        dynamic_position_1 = dynamic_picture_1.split('.')[0].split('_')
        dynamic_img_1 = np.array(
            Image.open('template_images/{}/{}/calibration/{}'.format(FT, template, dynamic_picture_1)).convert('RGB'))
        a_1 = img[int(int(dynamic_position_1[1]) - dynamic_img_1.shape[0] * scale):int(
            int(dynamic_position_1[1]) + dynamic_img_1.shape[0] * (1 + scale)),
              int(int(dynamic_position_1[0]) - dynamic_img_1.shape[1] * scale):int(
                  int(dynamic_position_1[0]) + dynamic_img_1.shape[1] * (1 + scale))]

        crop_x, crop_y = match_template_calibration_test1(
            'template_images/{}/{}/calibration/{}'.format(FT, template, dynamic_picture_1),
            a_1,
            scale=scale)
        print(crop_x, crop_y)
        a = get_area(img, 'template_images/{}/{}/text_area.txt'.format(FT, template), crop_x=crop_x, crop_y=crop_y,
                     scale=0)
        text = {}
        for key, value in a.items():
            if FT in large_FT and key == '大框_2':
                Image.fromarray(value).save('test.png')
                value = cv2.imread('test.png')
                B_channel, G_channel, R_channel = cv2.split(value)
                value = np.array(Image.fromarray(R_channel).convert('RGB'))
                if value.any() and value.shape[0] > 15:
                    images = text_predict(value)
                else:
                    continue
                v = []
                for i in images:
                    if i[1].any():
                        v.append([i[0], predict(i[1])[0]])
                new_v = link_same_line(v)
                mm = []
                mm, y1_y2 = get_big_text(new_v, FT, template)
                for i in mm:
                    text[i[0]] = i[1]
            else:
                text[key.split('_')[0]] = ''
                if key.split('_')[-1] == 'red':
                    Image.fromarray(value).save('test.png')
                    value = cv2.imread('test.png')
                    B_channel, G_channel, R_channel = cv2.split(value)
                    # _, RedThresh = cv2.threshold(R_channel, 160, 255, cv2.THRESH_BINARY)
                    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    # value = cv2.erode(RedThresh, element)
                    value = R_channel
                value = np.array(Image.fromarray(value).convert('RGB'))
                if value.any() and value.shape[0] > 15:
                    b = text_predict(value)
                else:
                    continue
                true_b = []
                new_b = []
                if b:
                    for i in b:
                        if i[1].shape[0] < i[1].shape[1]:
                            new_b.append(i)
                    if new_b:
                        max_y = sorted(new_b, key=lambda i: i[1].shape[0], reverse=True)[0][1].shape[0]
                        for i in new_b:
                            if i[1].shape[0] > 4 * max_y / 5:
                                true_b.append(i)
                        if key.split('_')[1] == '1':
                            true_b = sorted(true_b, key=lambda i: i[0][0])
                        elif key.split('_')[1] == '2':
                            true_b = sorted(true_b, key=lambda i: i[0][1])
                        for index, j in enumerate(true_b):
                            print(index, j[1].shape)
                            if j[1].any():
                                text[key.split('_')[0]] += predict(j[1])[0]
                                if key.split('/')[-1] == '金额':
                                    text[key.split('_')[0]] = money_transform(text[key.split('_')[0]])
                                if key.split('/')[-1] == '日期':
                                    text[key.split('_')[0]] = convert_time(text[key.split('_')[0]])
        print(text)
        text['证书名称'] = info[FT[:11]][1]
        num = 0
        text_clear = {key: value for key, value in text.items() if 'date' not in key}
        text_clear = {key: value for key, value in text_clear.items() if '所属主体名称' not in key}
        if FT.startswith('FT001003'):
            FT = 'FT001003109001'
        print(text_clear)
        for key, value in text_clear.items():
            if value == '':
                num += 1
        if num >= len(text_clear) / 2:
            return response.json({'result': 'false', 'message': '请求失败', 'taskid': task_id, 'fields': text, 'FT': FT})
        else:
            return response.json({'result': 'true', 'message': 'N/A', 'taskid': task_id, 'fields': text, 'FT': FT})
    except Exception as e:
        print(e)
        try:
            text = info[FT[:11]][0]
            if str(text) == 'nan':
                text = {}
            else:
                text = {i: '' for i in text.split('/')}
            return response.json({'result': 'false', 'message': '请求失败', 'taskid': task_id, 'fields': text, 'FT': FT})
        except Exception as e:
            return response.json({'result': 'false', 'message': '请求失败', 'taskid': task_id, 'fields': {}, 'FT': FT})


@app.route('/check')
def health_check(request):
    return response.text('ok')


if __name__ == '__main__':
    app.config.KEEP_ALIVE = False
    app.config.REQUEST_TIMEOUT = 900
    app.config.RESPONSE_TIMEOUT = 900
    app.run(host='0.0.0.0', port=8004)
