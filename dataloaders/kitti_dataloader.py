import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader
import cv2

class KITTIDataset(MyDataloader):
    def __init__(self, root, type, modality='d2sm'):
        super(KITTIDataset, self).__init__(root, type, modality)
        self.output_size = (352, 1216) 

    def train_transform(self, disp, depth, gt_depth, s_depth):
        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_np = depth / s
        disp_np = disp /s
        gt_depth_np = gt_depth/s
        s_depth_np = s_depth/s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        resize_out = (int(self.output_size[1]*s), int(self.output_size[0]*s))

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Rotate(angle),
            transforms.Resize(resize_out),
            transforms.BottomCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        depth_np = np.asfarray(depth_np, dtype='float32')
        depth_np = transform(depth_np)

        disp_np = np.asfarray(disp_np, dtype='float32')
        disp_np = transform(disp_np)

        gt_depth_np = np.asfarray(gt_depth_np, dtype='float32')
        gt_depth_np = transform(gt_depth_np)

        s_depth_np = np.asfarray(s_depth_np, dtype='float32')
        s_depth_np = transform(s_depth_np)

        return disp_np, depth_np, gt_depth_np, s_depth_np

    def val_transform(self, disp, depth, gt_depth, s_depth):
        depth_np = depth
        disp_np = disp
        gt_depth_np = gt_depth
        s_depth_np = s_depth
        transform = transforms.Compose([
            transforms.BottomCrop(self.output_size),
        ])
        depth_np = np.asfarray(depth_np, dtype='float32')
        depth_np = transform(depth_np)

        disp_np = np.asfarray(disp_np, dtype='float32')
        disp_np = transform(disp_np)

        gt_depth_np = np.asfarray(gt_depth_np, dtype='float32')
        gt_depth_np = transform(gt_depth_np)

        s_depth_np = np.asfarray(s_depth_np, dtype='float32')
        s_depth_np = transform(s_depth_np)

        return disp_np, depth_np, gt_depth_np, s_depth_np

