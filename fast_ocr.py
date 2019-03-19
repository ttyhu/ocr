
from pdf2image import convert_from_path
from ctpn.ctpn_blstm_test import text_predict
from densent_ocr.densenet_ocr_test import predict

import numpy as np
from PIL import Image

from sanic import Sanic, response

app = Sanic(__name__)


@app.route('/fce', methods=['POST'])
def fast_ocr(request):
    img_path = request.form.get('img_path')
    print(img_path)
    print(request.form.get('position'))
    position = '[' + request.form.get('position') + ']'
    page = request.form.get('pageNum', None)
    position = eval(position)
    if img_path.lower().endswith('pdf'):
        img = np.array(convert_from_path(img_path)[int(page) - 1])
    else:
        img = np.array(Image.open(img_path))
    crop_img = img[position[1]:position[3], position[0]:position[2]]
    crop_img = np.array(crop_img)
    crop_area = text_predict(crop_img)
    new_results = []
    for index, j in enumerate(crop_area):
        if j[1].any() and j[1].shape[0] < j[1].shape[1] * 1.5:
            try:
                # print(j[1].shape)
                new_results.append([j[0], predict(j[1])[0].replace('“', '').replace('‘', '')])
            except:
                continue
    document = ''
    new_results = sorted(new_results, key=lambda i: i[0][1])
    line_images = []
    cut_index = 0
    curr_index = 0
    for index, i in enumerate(new_results):
        if index == len(new_results) - 1:
            if cut_index < index:
                line_images.append(new_results[cut_index:index])
                line_images.append(new_results[index:])
            else:
                line_images.append(new_results[index:])
            break
        # if abs(new_results[index + 1][0][1] - new_results[index][0][1]) > (
        #         new_results[index][0][7] - new_results[index][0][1]) * 4 / 5:
        #     line_images.append(new_results[cut_index: index + 1])
        #     cut_index = index + 1
        if abs(new_results[index + 1][0][1] - new_results[curr_index][0][1]) > (
                new_results[curr_index][0][7] - new_results[curr_index][0][1]) * 4 / 5:
            line_images.append(new_results[cut_index: index + 1])
            cut_index = index + 1
            curr_index = index + 1
    for index, i in enumerate(line_images):
        line_images[index] = sorted(i, key=lambda a: a[0][0])
    texts = []
    for i in line_images:
        text = ''
        for index, j in enumerate(i):
            try:
                if index == len(i) - 1:
                    text += j[1]
                elif abs(i[index + 1][0][6] - i[index][0][6]) > 3 * (
                        abs(i[index][0][6] - i[index][0][0]) / len(i[index][1])):
                    text += j[1] + ' '
                else:
                    text += j[1]
            except:
                continue
        texts.append([i[0][0], text])
    for i in texts:
        document += i[1] + '\n'
    return response.text(document)


@app.route('/check')
def health_check(request):
    return response.text('ok')


if __name__ == '__main__':
    app.config.KEEP_ALIVE = False
    app.config.REQUEST_TIMEOUT = 900
    app.config.RESPONSE_TIMEOUT = 900
    app.run(host='0.0.0.0', port=8002)
