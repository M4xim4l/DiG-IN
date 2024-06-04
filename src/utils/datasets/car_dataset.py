""" Stanford Cars (Car) Dataset """
import os
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset


class CarDataset(Dataset):
    """
    # Description:
        Dataset for retrieving Stanford Cars images and labels

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
        self.num_classes = 196

        self.images = []
        self.targets = []
        self.path = path
        meta_mat = 'devkit/cars_meta.mat'
        meta_mat = loadmat(os.path.join(path, meta_mat))

        self.class_names = [class_name.item() for class_name in meta_mat['class_names'][0]]
        if phase == 'train':
            mat_file = 'devkit/cars_train_annos.mat'
            subdir = 'cars_train'
        else:
            mat_file = 'devkit/cars_test_annos_withlabels.mat'
            subdir = 'cars_test'

        list_path = os.path.join(path, mat_file)

        list_mat = loadmat(list_path)
        num_inst = len(list_mat['annotations']['fname'][0])
        for i in range(num_inst):
            path = list_mat['annotations']['fname'][0][i].item()
            path = os.path.join(subdir, path)
            label = list_mat['annotations']['class'][0][i].item()
            self.images.append(path)
            # count begin from zero
            self.targets.append(label - 1)

        print('Car Dataset with {} instances for {} phase'.format(len(self.images), self.phase))

        # transform
        self.transform = transform

    def __getitem__(self, item):
        # image
        image = Image.open(os.path.join(self.path, self.images[item])).convert('RGB')  # (C, H, W)
        if self.transform is not None:
            image = self.transform(image)

        # return image and label
        return image, self.targets[item]

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    ds = CarDataset('val', resize=[500,500])
    # print(len(ds))
    for i in range(0, 100):
        image, label = ds[i]
        # print(image.shape, label)
