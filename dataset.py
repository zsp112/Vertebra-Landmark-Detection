import os
import torch.utils.data as data
import pre_proc
import cv2
from scipy.io import loadmat
import numpy as np
import json

def rearrange_pts(pts):
    boxes = []
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k+4,:]
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:,1])
        y_inds_r = np.argsort(pt_r[:,1])
        tl = pt_l[y_inds_l[0], :]
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        # boxes.append([tl, tr, bl, br])
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(bl)
        boxes.append(br)
    return np.asarray(boxes, np.float32)

def loadPoint(annopath):
    # print(annopath)
    f = open(annopath   , 'r', encoding='utf-8')
    jsonData = json.load(f)
    pt = []
    pt_num = len(jsonData['shapes'])
    # for i in range(pt_num - 1, -1, -1):   # 倒序取点
    for i in range(0, pt_num, 1):     # 正序取点
        shape = jsonData['shapes'][i]
        pt.append(shape["points"][0])
    # for shape in jsonData['shapes']:
    #     pt.append(shape["points"][0])
    pt = np.asarray(pt)
    return pt

class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=4, vertebra_num=17):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.class_name = ['__background__', 'cell']
        self.num_classes = vertebra_num * 4
        self.vertebra_num = vertebra_num
        self.img_dir = os.path.join(data_dir, 'data', self.phase)
        self.img_ids = sorted(os.listdir(self.img_dir))

    def load_image(self, index):
        file_path = os.path.join(self.img_dir, self.img_ids[index])
        image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # image = cv2.imread(os.path.join(self.img_dir, self.img_ids[index]))
        return image

    def load_gt_pts(self, annopath):
        pts = loadmat(annopath)['p2']  # num x 2 (x,y)
        # 68 * 2
        pts = rearrange_pts(pts)
        return pts


    '''获得标注文件路径-原版'''
    def load_annoFolder(self, img_id):
        return os.path.join(self.data_dir, 'labels', self.phase, img_id + '.mat')

    def load_annotation(self, index):
        img_id = self.img_ids[index]
        prefix = img_id[:3]

        if prefix == 'sun':
            # 获得标注文件路径 mat
            annoFolder = self.load_annoFolder(img_id)
            pts = self.load_gt_pts(annoFolder)
        else:
            # json
            annoFolder = self.load_json_annoFolder(img_id)
            pts = self.load_json_gt_pts(annoFolder)
        return pts

    '''获得标注文件路径-json'''
    def load_json_annoFolder(self, img_id):
        img_id = img_id.split('.')[0]
        return os.path.join(self.data_dir, 'labels', self.phase, img_id + '.json')

    def load_json_gt_pts(self, annopath):
        # ----------------------------------------------------
        # 68 * 2
        pts = loadPoint(annopath)
        # ----------------------------------------------------
        pts = rearrange_pts(pts)
        return pts

    def load_my_annotation(self, index):
        img_id = self.img_ids[index]
        # 获得标注文件路径
        annoFolder = self.load_my_annoFolder(img_id)
        pts = self.load_my_gt_pts(annoFolder)
        return pts


    def __getitem__(self, index):
        img_id = self.img_ids[index]
        image = self.load_image(index)
        if self.phase == 'test':
            images = pre_proc.processing_test(image=image, input_h=self.input_h, input_w=self.input_w)
            return {'images': images, 'img_id': img_id}
        else:
            aug_label = False
            if self.phase == 'train':
                aug_label = True
            pts = self.load_annotation(index)   # num_obj x h x w
            out_image, pts_2 = pre_proc.processing_train(image=image,
                                                         pts=pts,
                                                         image_h=self.input_h,
                                                         image_w=self.input_w,
                                                         down_ratio=self.down_ratio,
                                                         aug_label=aug_label,
                                                         img_id=img_id)

            data_dict = pre_proc.generate_ground_truth(image=out_image,
                                                       pts_2=pts_2,
                                                       image_h=self.input_h//self.down_ratio,
                                                       image_w=self.input_w//self.down_ratio,
                                                       img_id=img_id)
            return data_dict

    def __len__(self):
        return len(self.img_ids)
