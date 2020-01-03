# -*- coding:utf-8 -*-
# @author :adolf
import json
from flask import Flask
from flask import request
import traceback
# from flask_cors import CORS
from inference import *
import base64
import numpy

"""
提供ocr服务
"""

app = Flask(__name__)


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


@app.route('/water_meter_service/', methods=["post", "get"], strict_slashes=False)
def service_main():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            image_with_base64 = in_dict['image']

            img = base64.b64decode(image_with_base64)
            img_array = np.frombuffer(img, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            img = img.astype(np.float32)
            detector = SegDetector()
            outp = detector.predict([img])
            print('========')
            print(outp)
            result_json = dict()
            result_json['result'] = outp
            return json.dumps(result_json, ensure_ascii=False,cls=MyEncoder)
        else:
            return json.dumps({"error_msg": "data is None", "status": 1}, ensure_ascii=False)


    except Exception as e:
        traceback.print_exc()
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2019, debug=True, processes=True)
