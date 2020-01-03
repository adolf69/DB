# -*- coding:utf-8 -*-
# @author :adolf
import cv2
import requests
import json
import base64
import numpy as np

from PIL import Image, ImageDraw, ImageFont


def get_result(encodestr):
    payload = {"image": encodestr, "type": "image"}
    r = requests.post("http://192.168.1.254:2019/water_meter_service/", json=payload)
    # print(r.text)
    res = json.loads(r.text)
    # print(res)
    return res


def put_text(img, text, left, top):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("仿宋_GB2312.ttf", 30, encoding="utf-8")
    draw.text((left, top - 30), text, (255, 0, 0), font=fontText)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img


def one_image(img_path):
    import os
    o_img = cv2.imread(img_path)

    with open(img_path, 'rb') as f:
        image = f.read()
        encodestr = str(base64.b64encode(image), 'utf-8')

    res_ = get_result(encodestr)
    return res_


def draw_img(img, box):
    # box = np.array(box, dtype=np.float)
    # box = np.array([[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]], np.int32)
    box = np.array(box)
    box = box.astype(np.int)

    box = box.reshape((-1, 1, 2))
    # print(box)
    # print(box.shape)

    cv2.polylines(img, [box], True, (0, 0, 255), 1)


if __name__ == '__main__':
    import time

    s1 = time.time()
    for i in range(1, 7):
        file_path = "/datadisk4/xinyi/water_meter/whole_images/whole{}.jpg".format(i)
        res = one_image(file_path)['result']['0']
        print(res)
        img = cv2.imread(file_path)
        h, w = img.shape[:2]
        print(h, w)
        for one_box in res:
            box = one_box['text_line']
            draw_img(img, box)
        cv2.imwrite('/datadisk4/xinyi/water_meter/whole_images/test_whole{}.jpg'.format(i), img)
