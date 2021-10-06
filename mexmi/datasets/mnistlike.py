import os.path as osp

from PIL import Image
from torch.utils.data.dataset import ChainDataset, IterableDataset, Dataset

from torchvision.datasets import MNIST as TVMNIST
from torchvision.datasets import EMNIST as TVEMNIST
from torchvision.datasets import FashionMNIST as TVFashionMNIST
from torchvision.datasets import KMNIST as TVKMNIST
from torchvision.datasets import MNIST_C as TVMNIST_C

import mexmi.config as cfg
import numpy as np

class MNIST(TVMNIST):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join(cfg.DATASET_ROOT, 'mnist')
        super().__init__(root, train, transform, target_transform, download)

# class CorruptedMNIST(TVMNIST):
#     def __init__(self, train=True, transform=None, target_transform=None, download=True):
#         root = osp.join(cfg.DATASET_ROOT, 'mnist')
#         super().__init__(root, train, transform, target_transform, download)
class CorruptedMNIST(TVMNIST_C):
    def __init__(self, train=True, transform=None, target_transform=None, download=True, corruption='fog'):
        root = osp.join(cfg.DATASET_ROOT, 'mnist_c')
        self.corruption = corruption#'spatter'#stripe'#'zigzag'#'spatter'#'rotate'#'motion_blur'#'glass_blur'#'fog'#'dotted_line'#'canny_edges'#'brightness'
        self._CORRUPTIONS = [
            'identity',
            'shot_noise',
            'impulse_noise',
            'glass_blur',
            'motion_blur',
            'shear',
            'scale',
            'rotate',
            'brightness',
            'translate',
            'stripe',
            'fog',
            'spatter',
            'dotted_line',
            'zigzag',
            'canny_edges',
        ]
        self._TRAIN_IMAGES_FILENAME = 'train_images.npy'
        self._TEST_IMAGES_FILENAME = 'test_images.npy'
        self. _TRAIN_LABELS_FILENAME = 'train_labels.npy'
        self._TEST_LABELS_FILENAME = 'test_labels.npy'
        super().__init__(root, train, transform, target_transform, download, self.corruption)

class KMNIST(TVKMNIST): #japan
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join(cfg.DATASET_ROOT, 'kmnist')
        super().__init__(root, train, transform, target_transform, download)


class EMNIST(TVEMNIST):
    def __init__(self, **kwargs):
        root = osp.join(cfg.DATASET_ROOT, 'emnist')
        super().__init__(root, split='balanced', download=True, **kwargs)
        # Images are transposed by default. Fix this.
        self.data = self.data.permute(0, 2, 1)

class EMNISTLetters(TVEMNIST):
    def __init__(self, **kwargs):
        root = osp.join(cfg.DATASET_ROOT, 'emnist')
        super().__init__(root, split='letters', download=True, **kwargs)
        # Images are transposed by default. Fix this.
        self.data = self.data.permute(0, 2, 1)


class FashionMNIST(TVFashionMNIST):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join(cfg.DATASET_ROOT, 'mnist_fashion')
        super().__init__(root, train, transform, target_transform, download)

class ChainMNIST(TVMNIST):
    #目前就提供三个以内的chain
    def __init__(self, chain=['MNIST','FashionMNIST','EMNIST'], train=True, transform=None, target_transform=None, download=True):
        super(ChainMNIST, self).__init__(root="", train=train, transform=transform, target_transform=target_transform, download=download)
        self.datasets = np.empty(len(chain)).astype('str').tolist() #['', '']
        self.len = np.zeros(len(chain)).tolist()
        self.chain = chain
        for i, c in enumerate(self.chain):
            if c.find('MNIST') == 0:
                if c.find("-") > -1:
                    Mlength = int(c.split("-")[1])
                else:
                    Mlength = 60000
                self.datasets[i] = MNIST(train=train, transform=transform)
                self.len[i] = Mlength#len(self.datasets[i])
            elif c == 'EMNIST':
                self.datasets[i] = EMNIST(train=train, transform=transform)
                self.len[i] = len(self.datasets[i])
            elif c == 'KMNIST':
                self.datasets[i] = KMNIST(train=train, transform=transform)
                self.len[i] = len(self.datasets[i])
            elif c == 'FashionMNIST':
                self.datasets[i] = FashionMNIST(train=train, transform=transform)
                self.len[i] = len(self.datasets[i])
            elif c == 'EMNISTLetters':
                self.datasets[i] = EMNISTLetters(train=train, transform=transform)
                self.len[i] = len(self.datasets[i])
            elif c.find('CorruptedMNIST') > -1:
                if c.find("-") > -1:
                    corruption = c.split("-")[1]
                    try:
                        mlength = int(c.split("-")[2])
                    except:
                        mlength = 60000
                else:
                    corruption = "fog"
                    mlength = 60000
                self.datasets[i] = CorruptedMNIST(train=train, transform=transform, corruption=corruption)
                self.len[i] = mlength #len(self.datasets[i])
        self.data = [self.datasets[i].data for i in range(len(self.chain))]
        self.targets = [self.datasets[i].targets for i in range(len(self.chain))]

    def getitemsinchain(self, idxs):# 来自numpy.random
        img_t = np.zeros(len(idxs)).tolist()
        target_t = np.zeros(len(idxs)).tolist()
        for i, idx in enumerate(idxs):
            if idx < self.len[0]:
                img_t[i], target_t[i] = self.data[0][idx], int(self.targets[0][idx])
            elif idx < self.len[0] + self.len[1]:
                idx = idx - self.len[0]
                img_t[i], target_t[i] = self.data[1][idx], int(self.targets[1][idx])
            else:
                idx = idx - self.len[0] - self.len[1]
                img_t[i], target_t[i] = self.data[2][idx], int(self.targets[2][idx])


        return img_t, target_t

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # if index < self.len[0]:
        #     img, target = self.datasets[0].__getitem__(index)
        # elif index < self.len[1]:
        #     idx = index - self.len[0]
        #     img, target = self.datasets[1].__getitem__(idx)
        # else:
        #     idx = index - self.len[0] - self.len[1]
        #     img, target = self.datasets[2].__getitem__(idx)

        if index < self.len[0]:
            img, target = self.data[0][index], int(self.targets[0][index])
            if self.chain[0].find('CorruptedMNIST') > -1:
                # print("self.chain[0]:", self.chain[0])
                img = Image.fromarray(np.asarray(img).squeeze(), mode='L')
            else:
                img = Image.fromarray(img.numpy(), mode='L')

        elif index < self.len[0] + self.len[1]:
            idx  = index - self.len[0]
            img, target = self.data[1][idx], int(self.targets[1][idx])
            if self.chain[1].find('CorruptedMNIST') > -1:
                # print("self.chain[0]:", self.chain[0])
                img = Image.fromarray(np.asarray(img).squeeze(), mode='L')
            else:
                img = Image.fromarray(img.numpy(), mode='L')
        else:
            idx = index - self.len[0] - self.len[1]
            img, target = self.data[2][idx], int(self.targets[2][idx])
            if self.chain[2].find('CorruptedMNIST') > -1:
                # print("self.chain[0]:", self.chain[0])
                img = Image.fromarray(np.asarray(img).squeeze(), mode='L')
            else:
                img = Image.fromarray(img.numpy(), mode='L')

        # # doing this so that it is consistent with all other datasets
        #
        # # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')
        # # np.asarray(img).squeeze()
        #
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    # def __iter__(self):
    #     for d in self.datasets:
    #         assert isinstance(d, Dataset), "ChainMNIST only supports Dataset"
    #         for x in d:
    #             yield x

    def __len__(self):
        total = 0
        for l in range(len(self.datasets)):
            # assert isinstance(d, Dataset), "ChainMNIST only supports Dataset"
            # total += len(d)
           total +=self.len[l]
        return total