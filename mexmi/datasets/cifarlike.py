import sys
import os
import os.path as osp

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets import CIFAR10 as TVCIFAR10
from torchvision.datasets import CIFAR100 as TVCIFAR100
from torchvision.datasets import SVHN as TVSVHN

# import datasets
import mexmi.config as cfg
from PIL import Image
from torchvision.datasets.utils import check_integrity
import pickle
import numpy as np

from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

#CIFARLike
# from datasets.downsampleimagenet import DownSampleImagenet32
class DownSampleImagenet32(VisionDataset):
    train_list = ['train_data_batch_{}'.format(i + 1) for i in range(10)]
    val_list = ['val_data']

    def __init__(self, root=os.path.join(cfg.DATASET_ROOT,'imagenet-32'), size=32, train=True, transform=None, classes=None):
        super().__init__(root, transform=transform)
        file_list = self.train_list if train else self.val_list
        self.data = []
        self.targets = []
        for filename in file_list:
            filename = os.path.join(self.root, 'train', filename) #directly put those files under the train folder
            with open(filename, 'rb') as f:
                entry = pickle.load(f)
            self.data.append(entry['data'].reshape(-1, 3, size, size))
            self.targets.append(entry['labels'])

        self.data = np.vstack(self.data).transpose((0, 2, 3, 1))
        self.targets = np.concatenate(self.targets).astype(int) - 1

        if classes is not None:
            classes = np.array(classes)
            filtered_data = []
            filtered_targets = []

            for l in classes:
                idxs = self.targets == l
                filtered_data.append(self.data[idxs])
                filtered_targets.append(self.targets[idxs])

            self.data = np.vstack(filtered_data)
            self.targets = np.concatenate(filtered_targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

class CIFAR10(TVCIFAR10):
    base_folder = 'cifar-10-batches-py'
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, train=True, transform=None, target_transform=None, download=False):
        root = osp.join(cfg.DATASET_ROOT, 'cifar10')
        super().__init__(root, train, transform, target_transform, download)

    def get_image(self, index):
        return self.data[index]


class CIFAR100(TVCIFAR100):
    base_folder = 'cifar-100-python'
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self, train=True, transform=None, target_transform=None, download=False):
        root = osp.join(cfg.DATASET_ROOT, 'cifar100')
        super().__init__(root, train, transform, target_transform, download)

    def get_image(self, index):
        return self.data[index]


class SVHN(TVSVHN):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join(cfg.DATASET_ROOT, 'svhn')
        # split argument should be one of {‘train’, ‘test’, ‘extra’}
        if isinstance(train, bool):
            split = 'train' if train else 'test'
        else:
            split = train
        self.classes = 10
        super().__init__(root, split, transform, target_transform, download)


class TinyImagesSubset(ImageFolder):
    """
    A 800K subset of the 80M TinyImages data consisting of 32x32 pixel images from the internet. 
    Note: that the dataset is unlabeled.
    """
    def __init__(self, train=True, transform=None, target_transform=None):
        root = osp.join(cfg.DATASET_ROOT, 'tiny-images-subset')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://github.com/Silent-Zebra/tiny-images-subset'
            ))

        # Initialize ImageFolder
        fold = 'train' if train else 'test'
        super().__init__(root=osp.join(root, fold), transform=transform,
                         target_transform=target_transform)
        self.root = root
        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',

                                                                len(self.samples)))
        self.chain = ['CIFAR10']

class ChainCIFAR(TVCIFAR10):
    #目前就提供三个以内的chain
    def __init__(self, chain=['CIFAR10', 'CIFAR100', "SVHN"], train=True, transform=None, target_transform=None, download=True):
        # super(ChainCIFAR, self).__init__(train=train, transform=transform, target_transform=target_transform, download=download)
        self.chain = chain
        self.datasets = np.zeros(len(chain)).tolist()
        self.len = np.zeros(len(chain)).tolist()
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []
        img_transform = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        for i, c in enumerate(chain):
            if c.find('CIFAR100') == 0:
                if c.find("-") > -1:
                    try:
                        clength = int(c.split("-")[1])
                    except:
                        clength = 50000
                else:
                    clength = 50000
                self.datasets[i] = CIFAR100(train=train, transform=transform, download=True)
                self.len[i] = clength
            elif c.find('CIFAR10') == 0:
                if c.find("-") > -1:
                    try:
                        clength = int(c.split("-")[1])
                    except:
                        clength = 50000
                else:
                    clength = 50000
                self.datasets[i] = CIFAR10(train=train, transform=transform, download=True)
                self.len[i] = clength
            elif c.find('SVHN') == 0:
                if c.find("-") > -1:
                    try:
                        clength = int(c.split("-")[1])
                    except:
                        clength = 73257
                else:
                    clength = 73257
                self.datasets[i] = SVHN(train=train, transform=transform, download=True)
                self.len[i] = clength
            elif c.find('TinyImagesSubset') == 0:
                if c.find("-") > -1:
                    try:
                        clength = int(c.split("-")[1])
                    except:
                        clength = 40000
                else:
                    clength = 40000
                self.datasets[i] = TinyImagesSubset(train=train, transform=transform)
                self.len[i] = clength

            elif c.find("DownSampleImagenet32") == 0:
                if c.find("-") > -1:
                    try:
                        clength = int(c.split("-")[1])
                    except:
                        clength = 1281167
                else:
                    clength = 1281167
                self.datasets[i] = DownSampleImagenet32(train=train, transform=img_transform)
                self.len[i] = clength
            if c.find('TinyImagesSubset') == 0:
                self.data.append(self.datasets[i].imgs)
            else:
                self.data.append(self.datasets[i].data)
            if c.find('SVHN') == 0:
                self.targets.append(self.datasets[i].labels)
            else:
                self.targets.append(self.datasets[i].targets)
        # self.data = [self.datasets[0].data, self.datasets[1].data]
        # self.targets = [self.datasets[0].targets, self.datasets[1].targets]#SVHN的label是target

    def getitemsinchain(self, idxs):  # 来自numpy.random
        img_t = np.zeros(len(idxs)).tolist()
        gt_label = np.zeros(len(idxs)).tolist()
        for i, idx in enumerate(idxs):
            if idx < self.len[0]:
                idx = int(idx)
                img_t[i], gt_label[i] = self.data[0][idx], int(self.targets[0][idx])

                if self.chain[0].find('SVHN') == 0:
                    img_t[i] = np.transpose(img_t[i], (1, 2, 0))
            elif idx < self.len[0] + self.len[1]:
                idx = int(idx - self.len[0])
                img_t[i], gt_label[i] = self.data[1][idx], int(self.targets[1][idx])

                if self.chain[1].find('SVHN') == 0:
                    img_t[i] = np.transpose(img_t[i], (1, 2, 0))
            else:
                idx = int(idx - self.len[0] - self.len[1])
                img_t[i], gt_label[i] = self.data[2][idx], int(self.targets[2][idx])

                if self.chain[2].find('SVHN') == 0:
                    img_t[i] = np.transpose(img_t[i], (1, 2, 0))

                # path, gt_label[i] = self.datasets[2].imgs[idx]
                # img_t[i] = self.datasets[2].loader(path)
                # img_t[i] = np.asarray(img_t[i])

        return img_t, gt_label

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # if index < self.len[0]:
        #     img, target = self.data[0][index], int(self.targets[0][index])
        # elif index < self.len[0] + self.len[1]:
        #     idx = index - self.len[0]
        #     img, target = self.data[1][idx], int(self.targets[1][idx])
        # else:
        #     idx = index - self.len[0] - self.len[1]
        #     img, target = self.data[2][idx], int(self.targets[2][idx])
        #     # path, target = self.datasets[2].imgs[idx]
        #     # img = np.asarray(self.datasets[2].loader(path))
        #
        # # doing this so that it is consistent with all other datasets
        # # to return a PIL Image
        # img = Image.fromarray(img)
        #
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        if index < self.len[0]:
            img, target = self.datasets[0][index]
        elif index < self.len[0] + self.len[1]:
            idx = index - self.len[0]
            img, target = self.datasets[1][idx]
        else:
            idx = index - self.len[0] - self.len[1]
            img, target = self.datasets[2][idx]

        return img, target

    def __len__(self):
        total = 0
        for d in self.len:#self.datasets:
            # assert isinstance(d, Dataset), "ChainMNIST only supports Dataset"
            total += d
        return total