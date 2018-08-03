from torchvision import transforms
import PIL.Image
import torch


def std_per_channel(images):
    images = torch.stack(images, dim = 0)
    return images.view(3, -1).std(dim = 1)


def mean_per_channel(images):
    images = torch.stack(images, dim = 0)
    return images.view(3, -1).mean(dim = 1)


class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im


class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __call__(self, tensor):
        tensor = (
            tensor - self.in_range[0]
        ) / (
            self.in_range[1] - self.in_range[0]
        ) * (
            self.out_range[1] - self.out_range[0]
        ) + self.out_range[0]
        return tensor


def make_transform(sz_resize = 256, sz_crop = 227, mean = [128, 117, 104], 
        std = [1, 1, 1], rgb_to_bgr = True, is_train = True, 
        intensity_scale = [[0, 1], [0, 255]]):
    return transforms.Compose([
        transforms.Compose([ # train: horizontal flip and random resized crop
            transforms.RandomResizedCrop(sz_crop),
            transforms.RandomHorizontalFlip(),
        ]) if is_train else transforms.Compose([ # test: else center crop
            transforms.Resize(sz_resize),
            transforms.CenterCrop(sz_crop),
        ]),
        transforms.ToTensor(),
        ScaleIntensities(
            *intensity_scale) if intensity_scale is not None else Identity(),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
        transforms.Lambda(
            lambda x: x[[2, 1, 0], ...]
        ) if rgb_to_bgr else Identity()
    ])

