import fitz

from PIL import Image
from pdf2image import convert_from_path
import numpy as np
from sanic import response, Sanic

from img_cls.predict_2 import predict_all


app = Sanic(__name__)


@app.route('/fre', methods=['POST'])
def img_cls(request):
    img_path = request.json.get('input')[0]
    par = request.json.get('par')
    page = request.json.get('page')
    print(img_path, page)
    try:
        if img_path.lower().endswith('.pdf'):
            pdf = fitz.open(img_path)  # 解析PDF
            page = pdf[int(page) - 1]  # 获得每一页的对象
            trans = fitz.Matrix(3, 3).preRotate(0)  # Matrix内参数为图像分辨率放大系数
            pm = page.getPixmap(matrix=trans, alpha=False)  # 获得每
            image = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            image.save('pi.jpg')
            image = np.array(Image.open('pi.jpg').convert('RGB').resize((299, 299)))
        else:
            image = np.array(Image.open(img_path).convert('RGB').resize((299, 299)))
        cls = predict_all(image, par)
    except Exception as e:
        print(e)
        return response.json({'result': 'false', 'Images': [], 'message': '请求失败'})
    return response.json({'result': 'true', 'Images': [{img_path: cls}], 'message': '请求成功'})


@app.route('/check')
def health_check(request):
    return response.text('ok')


if __name__ == '__main__':
    app.config.KEEP_ALIVE = False
    app.config.REQUEST_TIMEOUT = 900
    app.config.RESPONSE_TIMEOUT = 900
    app.run(host='0.0.0.0', port=8001)