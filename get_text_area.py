import pandas as pd


def get_area(img, text_area_file, crop_x, crop_y, scale):
    text_area = pd.read_csv(text_area_file, header=0, index_col=False)
    text_picture = {}
    for name in text_area.columns:
        x, y, w, h = text_area[name].values[0].split()
        img_text = img[(int(y) + int(crop_y)) - int(int(h) * scale):(int(y) + int(crop_y)) + int(int(h) * (1 + scale)),
                   int(x) + int(crop_x):int(x) + int(crop_x) + int(w)]
        text_picture[name] = img_text
    return text_picture


def get_area_none(text_area_file):
    text_area = pd.read_csv(text_area_file, header=0, index_col=False)
    text_picture = {}
    for name in text_area.columns:
        text_picture[name.split('_')[0]] = ''
    return text_picture
