# -*- coding:utf-8 -*-
# @author :adolf
import os
import cv2
import math
import pysnooper
import torch
import numpy as np
from concern.config import Config, Configurable
from shapely.geometry import Polygon

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
params = {
    'exp': 'experiments/seg_detector/rctw17_resnet50_deform_thre.yaml',
    'resume': '/model/seg_detector/final',
    'image_short_side': 2400,
    'image_long_side': 10000,
    'thresh': 0.3,
    'box_thresh': 0.1,
    'resize': True,
    'polygon': False,
    'verbose': False
}


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.deterministic = True


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
        model.load_state_dict(states, strict=True)
        print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if min(height, width) < self.args['image_short_side']:
            is_resize = True
            scale = self.args['image_short_side'] / min(height, width)
            if scale > 5:
                is_resize = False
            elif scale * max(height, width) > self.args['image_long_side']:
                is_resize = False
        else:
            is_resize = False
        # is_resize = False
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
                if scale == 0:
                    scale = 1
                new_image_short_side = scale * 32
                new_height = new_image_short_side
                new_width = int(math.ceil(new_height / height * width / 32) * 32)
                # if new_width == 0:
                #     new_width = 32
            else:
                scale = int(width / 32)
                if scale == 0:
                    scale = 1
                new_image_short_side = scale * 32
                new_width = new_image_short_side
                new_height = int(math.ceil(new_width / width * height / 32) * 32)
            # print(img.shape)
            # print(new_height, new_width)
            resized_img = cv2.resize(img, (new_width, new_height))
            # print(new_height, new_width)
        return resized_img

    def load_image(self, img):
        # print(img)
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img = img.astype(np.float32)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape

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
                    # print('*' * 50)
                    # print(i, j)
                    child_images.append((src_img[child_height * i: child_height * (i + 1) + extra_height,
                                         child_width * j: child_width * (j + 1) + extra_width, :],
                                         (child_height * i, child_width * j)))
            outputs[n] = child_images
        return outputs

    # def solve(self, box):
    #     """
    #     四点坐标转换为中心点+宽高
    #     """
    #     x1, y1, x2, y2, x3, y3, x4, y4 = box.flatten()[:8]
    #     cx = (x1 + x3 + x2 + x4) / 4.0
    #     cy = (y1 + y3 + y4 + y2) / 4.0
    #     w = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)) / 2
    #     h = (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)) / 2
    #
    #     if (x1 - x2) ** 2 + (y1 - y2) ** 2 == 0.0:
    #         sin12 = 0.0
    #     else:
    #         sin12 = (y2 - y1) / np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    #     if (x3 - x4) ** 2 + (y3 - y4) ** 2 == 0.0:
    #         sin34 = 0.0
    #     else:
    #         sin34 = (y3 - y4) / np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)
    #     angle12 = np.arcsin(sin12)
    #     angle34 = np.arcsin(sin34)
    #     angle = (angle12 + angle34) / 2
    #     if angle > 0:
    #         while angle > np.pi / 3:
    #             angle -= np.pi / 2
    #     else:
    #         while angle < -np.pi / 3:
    #             angle += np.pi / 2
    #     return angle, w, h, cx, cy
    #
    # def cut_image(self, box, image, left_adjust=3, right_adjust=3):
    #     angle, w, h, cx, cy = self.solve(box)
    #     left = max(1, int(cx - w / 2 - left_adjust))
    #     right = min(int(cx + w / 2 + right_adjust) + 1, image.shape[1] - 1)
    #     top = int(cy - h / 2)
    #     below = int(cy + h / 2) + 1
    #     image = image[top: below - 1, left: right, :]
    #     return image.numpy()

    def cut_image(self, box, image):
        # box [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8]

        left = max(1, int(min(box[:, 0]) - 2))
        right = min(int(max(box[:, 0]) + 2) + 1, image.shape[1] - 1)
        top = max(1, int(min(box[:, 1]) - 2))
        below = min(int(max(box[:, 1]) + 2) + 1, image.shape[0] - 1)
        image = image[top: below - 1, left: right, :]

        return image

    def order_points(self, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped

    def cut_image_v1(self, box, image):
        y1, x1, y2, x2 = np.min(box[0::2]), np.min(box[1::2]), np.max(box[0::2]), np.max(box[1::2])
        y_start = y1
        y_end = y2
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for i in range(-5, 5):
            index = max(y1 + i, 0)
            if sum(gray_img[index, :] < 240) / len(gray_img[index, :]) < 0.05:
                y_start = index
            index = min(y2 - i, gray_img.shape[0] - 1)
            if sum(gray_img[index, :] < 240) / len(gray_img[index, :]) < 0.05:
                y_end = index

        x_start = x1
        x_end = x2
        for i in range(-5, 5):
            index = max(x1 + i, 0)
            if sum(gray_img[index, :] < 240) / len(gray_img[index, :]) < 0.05:
                x_start = index
            index = min(x2 - i, gray_img.shape[1] - 1)
            if sum(gray_img[index, :] < 240) / len(gray_img[index, :]) < 0.05:
                x_end = index

        image = image[y_start: y_end - 1, x_start: x_end, :]
        return image

    def cut_image_v2(self, box, image):
        left = max(1, int(min(box[:, 0]) - 2))
        right = min(int(max(box[:, 0]) + 2) + 1, image.shape[1] - 1)
        top = max(1, int(min(box[:, 1]) - 2))
        below = min(int(max(box[:, 1]) + 2) + 1, image.shape[0] - 1)
        image = image[top: below - 1, left: right, :]
        image = image.astype(np.float32)
        h, w = image.shape[0], image.shape[1]
        # center = ((left + right) / 2, (top + below) / 2)
        pts1 = np.float32(
            [
                [box[0][0] - left, box[0][1] - top],
                [box[1][0] - left, box[1][1] - top],
                [box[3][0] - left, box[3][1] - top],
                # [box[2][0] - left, box[2][1] - top]
            ]
        )
        # print(666, [
        #     [box[3][0] - left, box[3][1] - top],
        #     [box[2][0] - left, box[2][1] - top],
        #     [box[0][0] - left, box[0][1] - top],
        #     [box[1][0] - left, box[1][1] - top],
        # ])
        pts2 = np.float32([
            [0, 0],
            [w, 0],
            [0, h],
            # [w, h]
        ])
        M = cv2.getAffineTransform(pts1, pts2)
        # M = cv2.getPerspectiveTransform(pts1, pts2)
        # rot_mat = cv2.getRotationMatrix2D(center, -0.4, 1)
        # img_rotated_by_alpha = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
        img_rotated_by_alpha = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))
        return img_rotated_by_alpha

    # 针对四边形进行padding
    def cut_image_v3(self, boxes, image):
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # _, image = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        left = max(1, int(min(boxes[:, 0])))
        right = min(int(max(boxes[:, 0])) + 1, image.shape[1] - 1)
        top = max(1, int(min(boxes[:, 1])))
        below = min(int(max(boxes[:, 1])) + 1, image.shape[0] - 1)
        # ori_img = image[top: below, left: right, :].copy()
        # cv2.imwrite('results_2/test2.png', image)
        copy_img = image.copy()
        ori_img = image.copy()

        boxes = np.array(boxes)
        boxes = boxes.astype(np.int)

        pad_col = np.mean(image[boxes[:, 1], boxes[:, 0], :])
        # print(pad_col)
        # print('======')
        # print(boxes)
        # cv2.imwrite('results_2/test1.png', ori_img)
        # part_image = image[top: below, left: right, :].copy()
        cv2.fillConvexPoly(copy_img, boxes, 0)

        mask_arr = np.where(copy_img != 0)
        ori_img[mask_arr] = pad_col
        part_image = ori_img[top: below, left: right, :]
        # pts = []
        # point_left_top = [boxes[0][0] - left, boxes[0][1] - top]
        # point_right_top = [(right - left) - (right - boxes[1][0]), boxes[1][1] - top]
        # point_left_down = [boxes[3][0] - left, (below - top) - (below - boxes[3][1])]
        # point_right_down = [(right - left) - (right - boxes[2][0]), (below - top) - (below - boxes[2][1])]
        # pts.append(point_left_top)
        # pts.append(point_right_top)
        # pts.append(point_left_down)
        # pts.append(point_right_down)
        # pts = np.array(pts)
        # part_image = self.four_point_transform(part_image, pts)
        return part_image

    @pysnooper.snoop("./ocr.log")
    def print_log(self, msg):
        print(msg)

    def predict(self, imgs, is_rectangle=True):
        outputs = {}
        # try:
        #     with futures.ThreadPoolExecutor(5) as executor:
        #         imgs_ = executor.map(do_rotate, imgs)
        #     imgs = list(imgs_)
        # except Exception as e:
        #     pass
        for i, img in enumerate(imgs):
            batch = dict()
            timg, original_shape = self.load_image(img)
            # print("1", "original_shape", original_shape)
            batch['shape'] = [original_shape]
            batch['image'] = timg
            print("1", "batch", batch)

            # self.model.eval()
            with torch.no_grad():
                # self.print_log("init") #14:51:24.794817
                pred = self.model.forward(batch, training=False)
                print("1", "pred:", pred)
                output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon'])
                text_lines, scores = output
                print("1", "text_lines:", text_lines)
                print("1", "scores:", scores)
                text_lines = text_lines[0]
                scores = scores[0]
                new_tls = []
                new_scs = []
                # self.print_log("step 1") #14:51:25.423924
                for text_line, score in zip(text_lines, scores):
                    # print(text_line.shape)
                    if len(text_line) == 0 or score < 0.2 or (not Polygon(text_line).is_valid) or Polygon(
                            text_line).area < 5:
                        print("1", "Polygon text_line", text_line)
                        print("1", "Polygon score", score)
                        continue
                    new_tls.append(text_line)
                    new_scs.append(score)
                # self.print_log("step 2") #14:51:25.439290
                new_tls = np.array(new_tls)
                new_scs = np.array(new_scs)
                print("1", "new_tls:", new_tls)
                print("1", "new_scs:", new_scs)
                # print('before connect:', new_tls.shape, new_scs.shape)
                # new_tls, new_scs = connect_quadrangles_by_distance(new_tls, new_scs)
                # print('after connect:', new_tls.shape, new_scs.shape)
                # new_tls, new_scs = cut_textlines(new_tls, new_scs, img)
                # print('after cut:', new_tls.shape, new_scs.shape)

                part_imgs = []
                # self.print_log("step 3") #14:51:26.622697
                for j, (text_line, score) in enumerate(zip(new_tls, new_scs)):
                    if is_rectangle:
                        cut_image = self.cut_image(text_line, img)
                        print("1", "cut_image", cut_image)
                        print("1", "cut_image_shape", cut_image.shape)
                    else:
                        try:
                            cut_image = self.cut_image_v3(text_line, img)
                            print("1", "not_rectangle cut_image", cut_image)
                            print("1", "not_rectangle cut_image_shape", cut_image.shape)
                        except:
                            cut_image = self.cut_image(text_line, img)
                            print("1", "not_rectangle cut_image", cut_image)
                            print("1", "not_rectangle cut_image_shape", cut_image.shape)
                    # cv2.imwrite("results_2/{}.jpg".format(j), cut_image)
                    part_imgs.append((cut_image, text_line.reshape(-1)))
                # self.print_log("step 4") #14:51:26.625548
                print("1", "part_imgs", part_imgs)
                outputs[i] = part_imgs
        return outputs
