#!python3
import argparse
import os
import torch
import cv2
import numpy as np
from experiment import Structure, Experiment
from concern.config import Configurable, Config
import math


def demo_visualize(image_path, output):
    boxes, _ = output
    boxes = boxes[0]
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_shape = original_image.shape
    pred_canvas = original_image.copy().astype(np.uint8)
    pred_canvas = cv2.resize(pred_canvas, (original_shape[1], original_shape[0]))

    for box in boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(pred_canvas, [box], True, (0, 0, 255), 1)

    return pred_canvas


def main():
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('exp', type=str)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--image_path', type=str, help='image path')
    parser.add_argument('--result_dir', type=str, default='./demo_results/', help='path to save results')
    parser.add_argument('--data', type=str,
                        help='The name of dataloader which will be evaluated on.')
    parser.add_argument('--image_short_side', type=int, default=736,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--thresh', type=float,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--box_thresh', type=float, default=0.6,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    parser.add_argument('--resize', action='store_true',
                        help='resize')
    parser.add_argument('--polygon', action='store_true',
                        help='output polygons if true')
    parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
                        help='Show iamges eagerly')
    parser.add_argument('--IsResize', action='store_true', help='is resize')

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    Demo(experiment, experiment_args, cmd=args).inference(args['image_path'], args['visualize'])


class Demo:
    def __init__(self, experiment, args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.args = cmd
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        self.model_path = self.args['resume']

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

    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img, is_resize=self.args['IsResize'])
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape

    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i, :, :].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")

    def pre_process(self, inputs, scale=1024, ratio=0.05):
        outputs = {}
        for n, src_img in enumerate(inputs):
            height, width = src_img.shape[:2]
            h_num = height / scale
            if h_num - int(h_num) < 0.2:
                h_num = max(int(h_num), 1)
            else:
                h_num = int(h_num) + 1
            if h_num == 1:
                outputs[n] = [(src_img, (0, 0))]
                continue
            w_num = width / scale
            if w_num - int(w_num) < 0.2:
                w_num = max(int(w_num), 1)
            else:
                w_num = int(w_num) + 1

            child_height = height // h_num + 1
            child_width = width // w_num + 1
            extra_height = int(child_height * ratio)
            extra_width = int(child_width * ratio)

            child_images = []
            for i in range(h_num):
                for j in range(w_num):
                    print('*' * 50)
                    print(i, j)
                    child_images.append((src_img[child_height * i: child_height * (i + 1) + extra_height,
                                         child_width * j: child_width * (j + 1) + extra_width, :],
                                         (child_height * i, child_width * j)))
            outputs[n] = child_images
        return outputs

    def inference(self, image_path, visualize=False):
        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)
        all_matircs = {}
        model.eval()
        batch = dict()
        batch['filename'] = [image_path]
        img, original_shape = self.load_image(image_path)
        # print('111', original_shape)

        batch['shape'] = [original_shape]
        batch['image'] = img

        with torch.no_grad():
            print(batch['image'])
            pred = model.forward(batch, training=False)
            print('pred', pred)

            output = self.structure.representer.represent(batch, pred,
                                                          is_output_polygon=self.args['polygon'])

            print('output', output)
            if not os.path.isdir(self.args['result_dir']):
                os.mkdir(self.args['result_dir'])
            self.format_output(batch, output)

            if visualize and self.structure.visualizer:
                # vis_image = self.structure.visualizer.demo_visualize(image_path, output)
                vis_image = demo_visualize(image_path, output)
                cv2.imwrite(os.path.join(self.args['result_dir'], image_path.split('/')[-1].split('.')[0] + '.jpg'),
                            vis_image)


if __name__ == '__main__':
    main()
