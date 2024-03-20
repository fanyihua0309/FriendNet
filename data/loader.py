import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img


def augment(imgs=[], size=256, edge_decay=0., data_augment=True):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    # simple re-weight for the edge
    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H - Hc)

    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W - Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    if data_augment:
        # horizontal flip
        if random.randint(0, 1) == 1:
            for i in range(len(imgs)):
                imgs[i] = np.flip(imgs[i], axis=1)

        # bad data augmentations for outdoor dehazing
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))

    return imgs


def align(imgs=[], size=256):
    H, W, _ = imgs[0].shape
    Hc, Wc = size, size

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]
    return imgs


class PairLoader(Dataset):
    def __init__(self, root_dir='data', mode='train', size=256, edge_decay=0, data_augment=True, cache_memory=False):
        assert mode in ['train', 'val', 'test']

        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.data_augment = data_augment

        self.cache_memory = cache_memory
        self.source_files = {}
        self.target_files = {}

        filename = f'{mode}.txt'
        txt_path = os.path.join(root_dir, filename)
        f = open(txt_path)
        lines = f.readlines()
        f.close()
        self.source_paths = []
        self.target_paths = []
        self.boxes = []
        for line in lines:
            source, target = line.split(' ')[0], line.split(' ')[1]
            self.source_paths.append(source)
            self.target_paths.append(target)
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line.split(' ')[2:]])
            self.boxes.append(box)
        self.img_num = len(self.target_paths)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        source_path = self.source_paths[idx]
        target_path = self.target_paths[idx]
        img_name = os.path.basename(target_path)
        box = self.boxes[idx]

        source_img = read_img(source_path) * 2 - 1
        target_img = read_img(target_path) * 2 - 1

        if self.mode != 'test':
            h, w, c = source_img.shape
            while h < self.size or w < self.size:
                idx = random.randint(0, len(self.source_paths) - 1)
                source_path = self.source_paths[idx]
                target_path = self.target_paths[idx]
                img_name = os.path.basename(target_path)
                source_img = read_img(source_path) * 2 - 1
                target_img = read_img(target_path) * 2 - 1
                h, w, c = source_img.shape

        ih, iw, _ = source_img.shape
        h, w = self.size, self.size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        box = np.array(box, dtype=np.float32)
        nL = len(box)
        labels_out = np.zeros((nL, 6))
        if nL:
            # ---------------------------------------------------#
            #   对真实框进行归一化，调整到0-1之间
            # ---------------------------------------------------#
            box[:, [0, 2]] = box[:, [0, 2]] / self.size
            box[:, [1, 3]] = box[:, [1, 3]] / self.size
            # ---------------------------------------------------#
            #   序号为0、1的部分，为真实框的中心
            #   序号为2、3的部分，为真实框的宽高
            #   序号为4的部分，为真实框的种类
            # ---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

            # ---------------------------------------------------#
            #   调整顺序，符合训练的格式
            #   labels_out中序号为0的部分在collect时处理
            # ---------------------------------------------------#
            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]

        # data augmentation
        if self.mode == 'train':
            [source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.data_augment)
            try:
                assert source_img.shape == (self.size, self.size, 3)
            except Exception:
                print(img_name, source_img.shape)

        elif self.mode == 'val':
            [source_img, target_img] = align([source_img, target_img], self.size)
            try:
                assert source_img.shape == (self.size, self.size, 3)
            except Exception:
                print(img_name, source_img.shape)

        elif self.mode == 'test':
            return {
                'hazy': hwc_to_chw(source_img),
                'clear': hwc_to_chw(target_img),
                'filename': img_name
            }

        return hwc_to_chw(source_img), hwc_to_chw(target_img), labels_out, img_name


# DataLoader 中 collate_fn 使用
def dataset_collate(batch):
    images = []
    clear_images = []
    bboxes = []
    filenames = []
    for i, (img, clear_img, box, filename) in enumerate(batch):
        images.append(img)
        clear_images.append(clear_img)
        box[:, 0] = i
        bboxes.append(box)
        filenames.append(filename)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    clear_images = torch.from_numpy(np.array(clear_images)).type(torch.FloatTensor)
    bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return {
        'hazy': images,
        'clear': clear_images,
        'detect_label': bboxes,
        'filename': filenames
    }

class SingleLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(self.root_dir))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_name = self.img_names[idx]
        img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

        return {'hazy': hwc_to_chw(img), 'filename': img_name}
