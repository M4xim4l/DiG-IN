""" FGVC Aircraft (Aircraft) Dataset
"""
import os
import pdb
from PIL import Image
from torch.utils.data import Dataset

FILENAME_LENGTH = 7


class AircraftDataset(Dataset):
    """
    # Description:
        Dataset for retrieving FGVC Aircraft images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, path, phase='train', transform=None):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.path = path
        variants_dict = {}
        with open(os.path.join(path, 'variants.txt'), 'r') as f:
            for idx, line in enumerate(f.readlines()):
                variants_dict[line.strip()] = idx
        self.num_classes = len(variants_dict)

        if phase == 'train':
            list_path = os.path.join(path, 'images_variant_trainval.txt')
        else:
            list_path = os.path.join(path, 'images_variant_test.txt')

        self.images = []
        self.labels = []
        with open(list_path, 'r') as f:
            for line in f.readlines():
                fname_and_variant = line.strip()
                self.images.append(fname_and_variant[:FILENAME_LENGTH])
                self.labels.append(variants_dict[fname_and_variant[FILENAME_LENGTH + 1:]])

        # transform
        self.transform = transform

    def __getitem__(self, item):
        # image
        image = Image.open(os.path.join(self.path, 'images', '%s.jpg' % self.images[item]))
        if self.transform is not None:
            image = self.transform(image)

        # return image and label
        return image, self.labels[item]  # count begin from zero

    def __len__(self):
        return len(self.images)

