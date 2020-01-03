# -*- coding:utf-8 -*-
# @author :adolf
# !python3
import os
import cv2
import math
import torch
import numpy as np
from concern.config import Config, Configurable
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "5"

params = {
    'exp': 'experiments/seg_detector/rctw17_resnet50_deform_thre.yaml',
    'resume': '/home/shizai/adolf/ai+rpa/ocr/ocr_use/DB/outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss_water/model/final',
    'image_short_side': 512,
    'thresh': 0.1,
    'box_thresh': 0.1,
    'resize': False,
    'polygon': False,
    'verbose': False
}


class SegDetector(object):
    def __init__(self, args=params):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])

        conf = Config()
        experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
        experiment_args.update(cmd=args)
        self.experiment = Configurable.construct_class_from_config(experiment_args)
        self.experiment.load('evaluation', **args)
        self.args = args
        self.structure = self.experiment.structure
        self.model_path = args['resume']

        self.init_torch_tensor()
        self.model = self.init_model()
        self.resume(self.model, self.model_path)
        self.model.eval()
        self.is_resize = args['resize']

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, img, is_resize=True):
        height, width, _ = img.shape
        if is_resize:
            if height < width:
                new_height = self.args['image_short_side']
                new_width = int(math.ceil(new_height / height * width / 32) * 32)
            else:
                new_width = self.args['image_short_side']
                new_height = int(math.ceil(new_width / width * height / 32) * 32)
            resized_img = cv2.resize(img, (new_width, new_height))
        else:
            if height < width:
                scale = int(height / 32)
                new_image_short_side = scale * 32
                new_height = new_image_short_side
                new_width = int(math.ceil(new_height / height * width / 32) * 32)
            else:
                scale = int(width / 32)
                new_image_short_side = scale * 32
                new_width = new_image_short_side
                new_height = int(math.ceil(new_width / width * height / 32) * 32)
            resized_img = cv2.resize(img, (new_width, new_height))
            print(new_height, new_width)
        return resized_img

    def load_image(self, img):
        original_shape = img.shape[:2]
        img = self.resize_image(img, is_resize=self.is_resize)
        img = img.astype(np.float32)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape

    def predict(self, imgs):
        outputs = dict()
        for i, img in enumerate(imgs):
            batch = dict()
            timg, original_shape = self.load_image(img)
            batch['shape'] = [original_shape]
            batch['image'] = timg

            with torch.no_grad():
                pred = self.model.forward(batch, training=False)
                output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon'])
                text_lines, scores = output
                text_lines = text_lines[0]
                scores = scores[0]
                new_tls = []
                new_scs = []
                for text_line, score in zip(text_lines, scores):
                    if len(text_line) == 0 or score < 0.2:
                        continue
                    new_tls.append(text_line)
                    new_scs.append(score)

                outputs[i] = []
                for j, (text_line, score) in enumerate(zip(new_tls, new_scs)):
                    one_output = dict()
                    one_output['text_line'] = text_line.tolist()
                    one_output['score'] = score
                    outputs[i].append(one_output)

                    # cut_image = self.cut_image(text_line, img)
                # cv2.imwrite("results_2/{}.jpg".format(j), cut_image)
                # part_imgs.append((cut_image, text_line.reshape(-1)))
                # outputs[i] = part_imgs
        return outputs


if __name__ == "__main__":
    # debug
    img = cv2.imread("images/2.png")
    img = img.astype(np.float32)
    detector = SegDetector()
    outp = detector.predict([img])
    print(outp)
