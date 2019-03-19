import os
from pdf2image import convert_from_path

from ctpn.ctpn_blstm_test import text_predict
from densent_ocr.model import predict
from extract_table import extract_table, generate_table
import cv2

import numpy as np
import docx
from PIL import Image
from docx.oxml.ns import qn
import time
from sanic import Sanic, response

app = Sanic(__name__)


def single_ocr(document, img_name):
    img_name.thumbnail((2000, 2000), Image.ANTIALIAS)
    img_name = img_name.convert('RGB')
    img = np.array(img_name)
    B_channel, G_channel, R_channel = cv2.split(img)
    cv2.imwrite('test.png', R_channel)
    img = cv2.cvtColor(R_channel, cv2.COLOR_GRAY2BGR)
    images = text_predict(img)
    # Image.fromarray(img).show()
    try:
        tables = extract_table(img)
        print(2222222222222222)
        has_table = True
    except:
        has_table = False
    results = []
    for index, j in enumerate(images):
        if j[1].any() and j[1].shape[0] < j[1].shape[1] * 1.5:
            try:
                if has_table:
                    count = 0
                    for table in tables:
                        if table[1][1] + table[1][3] > j[0][1] > table[1][1]:
                            continue
                        else:
                            count += 1
                    if count == len(tables):
                        content = predict(Image.fromarray(j[1]).convert('L'))
                        results.append([j[0], content.replace('“', '').replace('‘', '')])
                else:
                    content = predict(Image.fromarray(j[1]).convert('L'))
                    results.append([j[0], content.replace('“', '').replace('‘', '')])
            except Exception as e:
                continue
    results = sorted(results, key=lambda i: i[0][1])
    new_results = results
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
        for j in i:
            try:
                text += j[1]
            except:
                continue
        texts.append([i[0][0][1], text])
    max_x = [[i[0], i[1]] for i in texts]
    print(max_x)
    print(img_name.size)
    if has_table:
        for table in tables:
            table_index = 0
            for index, i in enumerate(texts):
                # print(i)
                # print(type(i[0]), type(table[1][1]))
                if i[0] == 'table':
                    if table[1][1] > i[1][1][1]:
                        table_index = index + 1
                elif table[1][1] > i[0]:
                    table_index = index + 1
            texts.insert(table_index, ['table', table])
    # print(texts)
    for i in texts:
        try:
            if i[0] == 'table':
                document = generate_table(document, i[1][0])
            else:
                document.add_paragraph(i[1])
        except:
            continue
    return document


@app.route('/full', methods=['POST'])
def full_ocr(request):
    print('start full ocr')
    start = time.time()
    pdf_file = list(request.json.get('input')[0].keys())[0]
    new_url = list(request.json.get('input')[0].values())[0]
    start_page = request.json.get('start_page')
    end_page = request.json.get('end_page')
    try:
        if pdf_file.lower().endswith('.pdf'):
            if start_page == end_page:
                pdf_images = [convert_from_path(pdf_file)[int(start_page)-1]]
            else:
                pdf_images = convert_from_path(pdf_file)[int(start_page)-1:int(end_page)]
        else:
            pdf_images = [Image.open(pdf_file)]
    except Exception as e:
        return response.json({'result': 'false', 'Documents': [], 'message': '请求失败'})
    document = docx.Document()
    document.styles['Normal'].font.name = u'宋体'
    document.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), u'宋体')
    for index, img_name in enumerate(pdf_images):
        print(index)
        if time.time() - start > 120 * len(pdf_images):
            return response.json({'result': 'false', 'Documents': [], 'message': '请求超时'})
        else:
            document = single_ocr(document, img_name)
    files = os.path.splitext(pdf_file)[0] + '.docx'
    new_url = new_url + files.split('/')[-1]
    document.save(new_url)
    return response.json({'result': 'true', 'Documents': [{'new_url': new_url}], 'message': '请求成功'})


@app.route('/check')
def health_check(request):
    return response.text('ok')


if __name__ == '__main__':
    app.config.KEEP_ALIVE = False
    app.config.RESPONSE_TIMEOUT = 7200
    app.run(host='0.0.0.0', port=8003)
