# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-7-15
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from randaugment import RandAugment
from .cutout import CutoutPIL, SLCutoutPIL
import torch
import os

logger = logging.getLogger(__name__)


class MLDataset(Dataset):
    def __init__(self, data_path, cfg, training=True):
        super(MLDataset, self).__init__()
        self.cfg = cfg
        self.training = training
        self.labels = [line.strip() for line in open(cfg.label_path)]
        self.num_classes = len(self.labels)
        self.label2id = {label:i for i, label in enumerate(self.labels)}

        self.data = []
        self.num_images = 0
        with open(data_path, 'r') as fr:
            for line in fr.readlines():
                img_path, img_label = line.strip().split('\t')
                img_label = [self.label2id[l] for l in img_label.split(',')]
                self.data.append([img_path, img_label])
                self.num_images = self.num_images + 1
        
        self.transform = self.get_transform() 
        logger.info(self.transform)

        # create the label mask
        self.mask = None
        # self.partial = 0.1
        if self.training == True and cfg.partial < 1.:
            # if not os.path.exists(cfg.label_mask_path):
            rand_tensor = torch.rand(self.num_images, self.num_classes)
            mask = (rand_tensor < cfg.partial).long()
            mask = torch.stack([mask], dim=1)
            torch.save(mask, os.path.join(cfg.org_path, cfg.data, 'partial_label_%.2f.pt' % cfg.partial))
            # else:
            #     mask = torch.load(cfg.label_mask_path)
            self.mask = mask.long()

    def get_transform(self):
        t = []
        t.append(transforms.Resize((self.cfg.img_size, self.cfg.img_size)))
        if self.training:
            t.append(transforms.RandomHorizontalFlip())
            # t.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)))
            t.append(CutoutPIL())  # default 黄块掩码噪声
            # t.append(SLCutoutPIL(self.cfg.n_holes, self.cfg.cutout_factor))
            t.append(RandAugment())
        t.append(transforms.ToTensor())
        if self.cfg.orid_norm:
            mean, std = [0, 0, 0], [1, 1, 1]
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        t.append(transforms.Normalize(mean=mean, std=std))
        
        return transforms.Compose(t)

    def __getitem__(self, index):
        img_path, img_label = self.data[index]
        img_data = Image.open(img_path).convert('RGB')
        img_data = self.transform(img_data)

        # one-hot encoding for label
        target = np.zeros(self.num_classes).astype(np.float32)
        target[img_label] = 1.0
        mask_label = 0
        # y_un = 0
        if self.mask is not None:
            masked = - torch.ones((1, self.num_classes), dtype=torch.long)
            target = self.mask[index] * target + (1 - self.mask[index]) * masked
            target = target.numpy().astype(np.float32)
            mask_label = self.mask[index]
            # y_bar = target+1
            # y_bar[y_bar == 1] = -1
            # y_bar[y_bar == 2] = 1
            # y_un = target + 0
            # y_un[y_un == -1] = 0.5

        item = {
            'img': img_data,
            'target': target,
            'img_path': img_path,
            'mask': mask_label,
            # 'y_sam': y_un,
            # 'y_fc': y_un,
        }
        
        return item
        
        
    def __len__(self):
        return len(self.data)