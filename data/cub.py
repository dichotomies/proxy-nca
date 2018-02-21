import os
from . import utils
import torch
import torchvision
import operator
import skimage
import numpy as np

def clean_folders(path):
    for path, dn, fns in os.walk(path): # e.g., 'cub200/images'
        for fn in fns:
            if fn[:2] == "._": # e.g., `._107.Common_Raven` (no image file)
                os.remove(os.path.join(path, fn))


class Birds(torch.utils.data.Dataset):

    def __init__(self, path, label_range):
        # e.g., label_range = [0, 50] for using first 50 classes only
        self.transform = utils.transform
        im_folder = torchvision.datasets.ImageFolder(root = path)
        # print(list(map(None, *im_folder.imgs)))
        # print(im_folder.imgs)
        # print([list(i) for i in zip(*(im_folder.imgs))])
        self.path_ims, self.ys = utils.select_by_label_range(
            # transpose list, e.g., 5x2 -> 2x5
            *[list(i) for i in zip(*(im_folder.imgs))], 
            label_range
        )
        # op = operator.ge if is_test else operator.lt
        # self.path_ims = [p for p, y in im_folder.imgs if op(y, 100)]
        # self.ys = [y for p, y in im_folder.imgs if op(y, 100)]


    def __len__(self):
        return len(self.ys)
        
    def __getitem__(self, index):
        im = skimage.io.imread(self.path_ims[index])
        if self.transform:
            im = self.transform(im)
        return im, self.ys[index]

