# -*- coding:utf-8 -*-
# @author :adolf
import json
from flask import Flask
from flask import request
import traceback
# from flask_cors import CORS
from service.service_inference import *
import base64
import numpy
import requests

"""
提供ocr服务
"""

app = Flask(__name__)


def rotate(image, angle, center=None, scale=1.0):
    h, w = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M_ = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M_, (w, h))

    rows, cols = rotated.shape[:2]

    for col in range(0, cols):
        if rotated[:, col].any():
            left = col
            break

    for col in range(cols - 1, 0, -1):
        if rotated[:, col].any():
            right = col
            break

    for row in range(0, rows):
        if rotated[row, :].any():
            up = row
            break

    for row in range(rows - 1, 0, -1):
        if rotated[row, :].any():
            down = row
            break

    res_widths = abs(right - left)
    res_heights = abs(down - up)
    res = np.zeros([res_heights, res_widths, 3], np.uint8)

    for res_width in range(res_widths):
        for res_height in range(res_heights):
            res[res_height, res_width] = rotated[up + res_height, left + res_width]

    return res


def get_result(encodestr):
    payload = {"image": encodestr, "type": "image"}
    r = requests.post("http://192.168.1.135:12125/rotate_service/", json=payload)
    # print(r.text)
    res = json.loads(r.text)
    # print(res)
    return res


# CORS(app, resources=r'/*')

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


@app.route('/gaoda_service/', methods=["post", "get"], strict_slashes=False)
def service_main():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            image_with_base64 = in_dict['image']

            rotate_direction = get_result(image_with_base64)['result']

            img = base64.b64decode(image_with_base64)
            img_array = np.frombuffer(img, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if rotate_direction == 1:
                img = rotate(img, 270)
            elif rotate_direction == 2:
                img = rotate(img, 180)
            elif rotate_direction == 3:
                img = rotate(img, 90)

            img = img.astype(np.float32)
            detector = SegDetector()
            outp = detector.predict([img])
            print('========')
            print(outp)
            result_json = dict()
            result_json['result'] = outp
            return json.dumps(result_json, ensure_ascii=False, cls=MyEncoder)
        else:
            return json.dumps({"error_msg": "data is None", "status": 1}, ensure_ascii=False)


    except Exception as e:
        traceback.print_exc()
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=12020, debug=True, processes=True)
