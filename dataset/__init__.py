
from .cars import Cars
from .cub import CUBirds
from . import utils
from .base import BaseDataset


_type = {
    'cars': Cars,
    'cub': CUBirds
}


def load(name, root, classes, transform = None):
    return _type[name](root = root, classes = classes, transform = transform)
    
