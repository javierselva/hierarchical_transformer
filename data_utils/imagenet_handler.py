from torchvision.datasets import ImageFolder
from torchvision.transforms import RandomCrop, CenterCrop, PILToTensor, Compose
import os

# TODO this class could/should be a function
class CustomDataset(ImageFolder):
    def __init__(self, config, mode, transform=None, **kwargs):
        params = config['data']['imagenet']['sampler']
        if transform is None:
            transforms = [PILToTensor()]
            crop_size = (params['height'], params['width'])
            if mode == 'train':
                transforms.append(RandomCrop(crop_size))
            elif mode == 'val':
                transforms.append(CenterCrop(crop_size))
            else:
                transforms.append(CenterCrop(crop_size))
            transform = Compose(transforms)
        path = config['data']['imagenet']['location']
        super(CustomDataset).__init__(os.path.join(path, mode), transform=transform, **kwargs)


