from __future__ import print_function

import numpy as np
from skimage import color

import torchvision.datasets as datasets


class ImageFolderInstance(datasets.ImageFolder):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index)
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img):

        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        return img
