import os
import random
from glob import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import itertools
from torch.utils.data.sampler import Sampler


def _list_images_masks(root, split):
    """
    针对 OCTA 数据集的预期目录结构:
      root/
        train/
          images/xxx.bmp
          masks/xxx.bmp
        val/
          images/xxx.bmp
          masks/xxx.bmp
        test/
          images/xxx.bmp
          masks/xxx.bmp
    """
    img_dir = os.path.join(root, split, "images")
    msk_dir = os.path.join(root, split, "masks")

    # 优先匹配 .bmp 文件，过滤掉系统隐藏文件
    imgs = sorted(glob(os.path.join(img_dir, "*.bmp")))
    msks = sorted(glob(os.path.join(msk_dir, "*.bmp")))

    # 容错机制：如果没找到 bmp，退回到匹配所有文件
    if len(imgs) == 0:
        imgs = sorted(glob(os.path.join(img_dir, "*")))
        msks = sorted(glob(os.path.join(msk_dir, "*")))
        # 过滤掉可能的文件夹本身
        imgs = [f for f in imgs if os.path.isfile(f)]
        msks = [f for f in msks if os.path.isfile(f)]

    assert len(imgs) == len(msks), f"images({len(imgs)}) != masks({len(msks)}) in {split}"
    return imgs, msks


class OCTADataset(Dataset):
    def __init__(self, root, split="train", image_size=(512, 512), transform=None):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.images, self.masks = _list_images_masks(root, split)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        msk_path = self.masks[idx]

        # 哪怕是灰度图也转成 RGB，满足 SAM 编码器的 3 通道要求
        img = Image.open(img_path).convert("RGB").resize(self.image_size, resample=Image.BILINEAR)
        # 掩码转成单通道
        msk = Image.open(msk_path).convert("L").resize(self.image_size, resample=Image.NEAREST)

        img = np.asarray(img).astype(np.float32) / 255.0  # H,W,3
        msk = (np.asarray(msk) > 127).astype(np.uint8)  # 二值化为 0 和 1

        sample = {"image": img, "label": msk}

        if self.transform is not None:
            sample = self.transform(sample)

        # to tensor
        image = torch.from_numpy(sample["image"].transpose(2, 0, 1)).float()  # 3,H,W
        label = torch.from_numpy(sample["label"]).long()  # H,W (0/1)
        return {"image": image, "label": label}


class RandomRotFlip2D:
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # image: H,W,3 ; label: H,W
        k = random.randint(0, 3)
        image = np.rot90(image, k).copy()
        label = np.rot90(label, k).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            label = np.flip(label, axis=0).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        return {"image": image, "label": label}


class RandomColorJitter:
    """Very light jitter without torchvision dependency in dataset."""

    def __init__(self, brightness=0.1, contrast=0.1):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        b = 1.0 + random.uniform(-self.brightness, self.brightness)
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        mean = image.mean(axis=(0, 1), keepdims=True)
        image = (image - mean) * c + mean
        image = image * b
        image = np.clip(image, 0.0, 1.0)
        return {"image": image, "label": label}


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class TwoStreamBatchSampler(Sampler):
    """
    Same logic as your LAHeart TwoStreamBatchSampler, reused for 2D.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = np.random.permutation(self.primary_indices)

        # itertools trick above doesn't terminate; use a generator:
        def infinite_shuffles():
            while True:
                yield np.random.permutation(self.secondary_indices)

        secondary_iter = itertools.chain.from_iterable(infinite_shuffles())

        def grouper(it, n):
            args = [iter(it)] * n
            return zip(*args)

        return (
            list(primary_batch) + list(secondary_batch)
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size