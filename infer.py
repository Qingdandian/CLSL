import os
import yaml
import warnings
warnings.filterwarnings("ignore")
import argparse
from tqdm import tqdm
import cv2
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader

from models.factory import create_model
from lib.util import *
from lib.metrics import *
from lib.dataset import MLDataset

torch.backends.cudnn.benchmark = True


class Inference(object):
    def __init__(self, cfg):
        super().__init__()
        # dataset = MLDataset(cfg.test_path, cfg, training=False)
        dataset = MLDataset(cfg.train_path, cfg, training=False)
        self.labels = dataset.labels
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        
        self.model = create_model(cfg.model, cfg=cfg)
        self.model.cuda()
        self.ema_model = ModelEma(self.model, decay=cfg.ema_decay)

        self.cfg = cfg

    @torch.no_grad()
    def run(self):
        model_dict = torch.load(self.cfg.ckpt_ema_best_path)
        self.ema_model.load_state_dict(model_dict)
        print(f'loading best checkpoint success')
        
        score_list, target_list, more_target_list = [], [], []
        self.model.eval()
        for batch in tqdm(self.dataloader):
            img = batch['img'].cuda()
            target = batch['target'].numpy()[0]
            img_path = batch['img_path'][0]
            # COCO
            # if 'COCO_val2014_000000574769' not in img_path and 'COCO_val2014_000000540172' not in img_path and \
            #     'COCO_val2014_000000000241' not in img_path and 'COCO_val2014_000000002431' not in img_path and \
            #     'COCO_val2014_000000002477' not in img_path and 'COCO_val2014_000000003091' not in img_path and \
            #     'COCO_val2014_000000006608' not in img_path and 'COCO_val2014_000000009527' not in img_path:
            #     continue
            # VOC
            if '000710' not in img_path :
                continue
            # if '008895' not in img_path:
            #     continue
            
            ret = self.ema_model(img)
            # score = torch.sigmoid(ret['logits']).cpu().numpy()
            score = ret['logits'].cpu().numpy()
            score_copy = score.copy()
            score_list.append(score)
            # att_weights = ret['alpha'].reshape(1, 20, 14, 14).squeeze().cpu().numpy()
            att_weights = (torch.softmax(ret['alpha'], dim=-1)).reshape(1, 20, 14, 14).squeeze().cpu().numpy()
            target_list.append(target)
            score_copy[score_copy >= 0.5] = 1
            score_copy[score_copy < 0.5] = 0
            more_target = score_copy.squeeze()
            more_target = [extra - gt for extra, gt in zip(more_target, target)]
            more_target = np.asarray(more_target)
            more_target[more_target < 0] = 0
            # more_target_list.append(more_target)

            img_name = os.path.basename(img_path).rsplit('.', 1)[0]
            save_dir = os.path.join(self.cfg.exp_dir, 'visualization', img_name)
            check_makedir(save_dir)
            self.att_weight_visualization(att_weights, target, img_path, save_dir)
            self.more_att_weight_visualization(att_weights, more_target, img_path, save_dir)
            
        scores = np.concatenate(score_list, axis=0)
        targets = np.stack(target_list, axis=0)
        np.save(os.path.join(self.cfg.exp_dir, 'scores.npy'), scores)
        np.save(os.path.join(self.cfg.exp_dir, 'targets.npy'), targets)
        
    def att_weight_visualization(self, att_weights, target, img_path, save_dir, intensity=0.5):
        gt_labelids = np.nonzero(target)[0]
        gt_labels = [self.labels[i] for i in gt_labelids]
        # 'E:/做实验时用到的数据集/voc/datasets/VOC2007\\JPEGImages\\000001.jpg'
        # img_data = cv2.imread(img_path)
        img_data = cv2.imread('E:/infer_data\\000710.jpg')
        img_data = cv2.resize(img_data, (self.cfg.img_size, self.cfg.img_size))
        cv2.imwrite(os.path.join(save_dir, 'raw_image.jpg'), img_data)
        
        t = 1 / (att_weights.shape[-1] * att_weights.shape[-2])
        for label, weight in zip(self.labels, att_weights):
            if label in gt_labels:
                if np.max(weight) - np.min(weight) > t:
                    weight = weight - np.min(weight)
                    weight = weight / np.max(weight)
                weight = cv2.resize(weight, (self.cfg.img_size, self.cfg.img_size))
                heatmap = cv2.applyColorMap(np.uint8(255*weight), cv2.COLORMAP_JET)
                att_image = heatmap * intensity + img_data
                
                name = '_{}_a3.jpg'
                cv2.imwrite(os.path.join(save_dir, name.format(label)), att_image)

    def more_att_weight_visualization(self, att_weights, target, img_path, save_dir, intensity=0.5):
        gt_labelids = np.nonzero(target)[0]
        gt_labels = [self.labels[i] for i in gt_labelids]
        # 'E:/做实验时用到的数据集/voc/datasets/VOC2007\\JPEGImages\\000001.jpg'
        # img_data = cv2.imread(img_path)
        img_data = cv2.imread('E:/infer_data\\000710.jpg')
        img_data = cv2.resize(img_data, (self.cfg.img_size, self.cfg.img_size))
        cv2.imwrite(os.path.join(save_dir, 'raw_image.jpg'), img_data)

        t = 1 / (att_weights.shape[-1] * att_weights.shape[-2])
        for label, weight in zip(self.labels, att_weights):
            if label in gt_labels:
                if np.max(weight) - np.min(weight) > t:
                    weight = weight - np.min(weight)
                    weight = weight / np.max(weight)
                weight = cv2.resize(weight, (self.cfg.img_size, self.cfg.img_size))
                heatmap = cv2.applyColorMap(np.uint8(255 * weight), cv2.COLORMAP_JET)
                att_image = heatmap * intensity + img_data

                name = '_extra_{}_a3.jpg'
                cv2.imwrite(os.path.join(save_dir, name.format(label)), att_image)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, default='experiments/sarlp_voc2007/exp58')  # experiments/mlic_v3_voc2007/exp91   experiments/mlic_voc2007/exp10
    args = parser.parse_args()
    cfg_path = os.path.join(args.exp_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError('config file not found in the {}!'.format(cfg_path))
    cfg = yaml.load(open(cfg_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    cfg = argparse.Namespace(**cfg)
    print(cfg)

    infer = Inference(cfg)
    infer.run()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--exp-dir', type=str,
    #                     default='experiments/mlic_voc2007/exp10')  # experiments/mlic_v3_voc2007/exp91    experiments/mlic_voc2007/exp10
    # args = parser.parse_args()
    # cfg_path = os.path.join(args.exp_dir, 'config.yaml')
    # if not os.path.exists(cfg_path):
    #     raise FileNotFoundError('config file not found in the {}!'.format(cfg_path))
    # cfg = yaml.load(open(cfg_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    # cfg = argparse.Namespace(**cfg)
    # print(cfg)
    #
    # infer = Inference(cfg)
    # infer.run()

