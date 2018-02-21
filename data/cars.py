import scipy.io
import skimage.io
import torch
from . import utils
import os

class Cars(torch.utils.data.Dataset):
    
    def filter_images(self):
        # some images are gray-scale, filter them out 

        paths, ys = [], []
        for y, path in zip(self.ys, self.paths):

            im = skimage.io.imread(
                os.path.join(
                    self.path_ims,
                    path 
                )
            )
            
            if len(im.shape) == 3:
                # add only 3-channel images
                paths.append(path)
                ys.append(y)

        self.paths = paths
        self.ys = ys
    
    def __init__(self, 
        path_ims, path_annotations, label_range, transform = utils.transform
    ):
        
        cars = scipy.io.loadmat(path_annotations)
        self.path_ims = path_ims
        self.transform = transform
        ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]
        # remove `car_ims`, e.g. `car_ims/000046.jpg` to `000046.jpg`
        im_paths = [a[0][0].split("/")[1] for a in cars['annotations'][0]]
        self.paths, self.ys = utils.select_by_label_range(
            im_paths, ys, label_range
        )
        # flatten ys
        self.filter_images()
        assert len(self.paths) == len(self.ys)


    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = skimage.io.imread(
            os.path.join(
                self.path_ims,
                self.paths[index] 
            )
        )
        
        if self.transform:
            im = self.transform(im)
        return im, self.ys[index]
