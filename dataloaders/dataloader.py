import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py
import dataloaders.transforms as transforms
import torch
import cv2

IMG_EXTENSIONS = ['.h5',]

kernel = np.array([[0.06217652, 0.24935221, 0.06217652],
 [0.24935221, 1.        , 0.24935221],
 [0.06217652, 0.24935221, 0.06217652]])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

def h5_loader(path):
    # read h5 data
    h5f = h5py.File(path, "r")

    # depth completed by self-supervised completion
    depth = np.array(h5f['depth_c'], dtype=np.float32)/255

    # depth from PSMNet
    disp = np.array(h5f['disp_c'], dtype=np.float32)/255

    # sparse depth by lidar
    s_depth = np.array(h5f['D'], dtype=np.float32)/255

    # semi-dense points as the groundtruth
    gt_depth = np.array(h5f['D_semi'], dtype=np.float32)/255
    return disp, depth, gt_depth, s_depth

to_tensor = transforms.ToTensor()

class MyDataloader(data.Dataset):
    modality_names = ['d2sm'] # , 'g', 'gd'
    

    def __init__(self, root, type, sparsifier=None, modality='d2sm', loader=h5_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        assert len(imgs)>0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        if type == 'train':
            self.transform = self.train_transform
        elif type == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))
        self.loader = loader
        self.sparsifier = sparsifier

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

    def train_transform(self, disp, depth, gt_depth, s_depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(rgb, disp, depth, gt_depth, s_depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def create_mask(self, s_depth):
        return s_depth > 0
    def dilate_mask(self, mask):
        mask_dilate = np.asarray(mask.copy(), dtype = np.float32)
        mask_dilate = cv2.filter2D(mask_dilate, -1, kernel)
        mask_dilate[mask_dilate > 1.0001]=1.
        return mask_dilate

    def create_d2sm(self, disp, depth, s_depth):
        mask = self.create_mask(s_depth)
        mask_dilate = self.dilate_mask(mask)
        d2 = np.append(np.expand_dims(disp, axis=2), np.expand_dims(depth, axis=2),axis=2)
        d2s = np.append(d2, np.expand_dims(s_depth, axis=2),axis=2)
        d2sm = np.append(d2s, np.expand_dims(mask, axis=2), axis=2)
        d2sm = np.append(d2sm, np.expand_dims(mask_dilate, axis=2), axis=2)
        return d2sm

    def __getraw__(self, index):
        path, target = self.imgs[index]
        #rgb, depth = self.loader(path)
        disp, depth, gt_depth, s_depth = self.loader(path)
        return disp, depth, gt_depth, s_depth

    def __getitem__(self, index):
        disp, depth, gt_depth, s_depth = self.__getraw__(index)
        if self.transform is not None:
            disp_np, depth_np, gt_depth_np, s_depth_np = self.transform(disp, depth, gt_depth, s_depth)
        else:
            raise(RuntimeError("transform not defined"))

        if self.modality == 'd2sm':
            input_np = self.create_d2sm(disp_np, depth_np, s_depth_np)

        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        gt_depth_tensor = to_tensor(gt_depth_np)
        gt_depth_tensor = gt_depth_tensor.unsqueeze(0)
    
        return input_tensor, gt_depth_tensor

    def __len__(self):
        return len(self.imgs)