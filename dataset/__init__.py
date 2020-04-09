
from .cars import Cars
from .cub import CUBirds
from .sop import SOProducts
from . import utils
from .base import BaseDataset


_type = {
    'cars': Cars,
    'cub': CUBirds,
    'sop': SOProducts
}


def load(name, root, classes, transform = None):
    return _type[name](root = root, classes = classes, transform = transform)
    
