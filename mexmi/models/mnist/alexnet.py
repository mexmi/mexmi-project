'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei
'''
import torch.nn as nn

__all__ = ['alexnet']

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # self.classifier = nn.Linear(256, num_classes)
        self.fc6 = nn.Linear(256 * 3 * 3, 1024)  # AlexFC6(256*6*6, 4096)
        self.fc7 = nn.Linear(1024, 512)  # AlexFC6(4096,4096)
        self.fc8 = nn.Linear(512, 10)  # AlexFC6(4096,1000)

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = x.view(-1, 256 * 3 * 3)
        # x = self.classifier(x)
        x = self.fc6(x)
        x = nn.ReLU(x)
        x = self.fc7(x)
        x = nn.ReLU(x)
        x = self.fc8(x)
        return x


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model
