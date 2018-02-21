import torchvision as tv

def select_by_label_range(zs, ys, r): # zs : images or paths, corr. to ys
    assert len(zs) == len(ys)
    zs = [z for z, y in zip(zs, ys) if r[0] <= y and y < r[1]]
    ys = [y - r[0] for y in ys if r[0] <= y and y < r[1]]
    return zs, ys

transformations = [
    tv.transforms.ToPILImage(),
    tv.transforms.Resize(340),
    tv.transforms.RandomCrop(299),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
]

transform = tv.transforms.Compose(transformations)
