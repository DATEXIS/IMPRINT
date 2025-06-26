"""
Adapted from the source
https://github.com/liyxi/mnist-m/blob/main/mnist_m.py
"""

import os
import warnings

import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive


class MNISTM(VisionDataset):
    """MNIST-M Dataset."""

    resources = {
        "train": (
            "https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz",
            "191ed53db9933bd85cc9700558847391",
        ),
        "test": (
            "https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz",
            "e11cb4d7fff76d7ec588b1134907db59",
        ),
    }

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if download:
            self.download()

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets = torch.load(os.path.join(self.root, data_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "processed")

    def download(self):
        """Download the desired MNIST-M data (train or test)."""
        split_str = "train" if self.train else "test"
        url, md5 = self.resources[split_str]
        filename = url.rpartition("/")[2]
        download_and_extract_archive(
            url,
            download_root=self.root,
            extract_root=self.root,
            filename=filename,
            md5=md5,
        )

        print("Done!")

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
