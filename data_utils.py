
import os
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils import data
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from tqdm.notebook import tqdm

from histaugan.model import MD_multi

# ------------
# dataset classes for the usage with patches in hdf5 files
# ------------

class Center0Dataset(data.Dataset):
    """Adapted from https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
    Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(self, file_path, phase, balanced=False, load_data=True, data_cache_size=3, transform=transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform
        assert phase in [
            'train', 'val'], 'phase is not valid. should be either train or val'
        self.phase = phase

        if self.phase == 'val':
            self.slides = ['patient_015_node_1', 'patient_015_node_2']
        else:
            self.slides = ['patient_004_node_4', 'patient_009_node_1', 'patient_010_node_4', 'patient_012_node_0',
                           'patient_016_node_1', 'patient_017_node_1', 'patient_017_node_2', 'patient_017_node_4']

        self.data = {
            'patches': [],
            'tumor_ratio': [],
        }

        assert os.path.isfile(file_path)

        with h5py.File(file_path, 'r') as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                if gname in self.slides:
                    for dname, ds in group.items():
                        if dname == 'coordinates':
                            continue
                        else:
                            self.data[dname].append(torch.from_numpy(ds[:]))
                else:
                    continue

        self.data['patches'] = torch.cat(
            self.data['patches']).permute(0, 3, 1, 2)
        self.data['tumor_ratio'] = torch.cat(
            self.data['tumor_ratio']).unsqueeze(-1)

    def __getitem__(self, index):  # ok
        # get data
        x = self.data['patches'][index]  # data is stored in ByteTensors
        x = x.float().div_(255.)
        if self.transform:
            x = self.transform(x)

        # get label
        y = self.data['tumor_ratio'][index]
        # set labels larger than 0 to 1, i.e. tumor positive
        y = (y >= 0.01).float()
        return x, y

    def __len__(self):
        return len(self.data['patches'])


class OneCenterLoad(data.Dataset):
    """Adapted from https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
    Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(self, data_dir, center, phase, transform=transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), norm=None):
        super().__init__()
        self.data_dir = data_dir
        self.center = center
        assert phase in ['train', 'val'], 'phase is not valid. should be either train or val'
        self.phase = phase
        self.transform = transform
        self.norm = norm

        self.val_slides = [
            'patient_015_node_1', 'patient_015_node_2',
            'patient_020_node_2', 'patient_020_node_4',
            'patient_046_node_3', 'patient_046_node_4',
            'patient_075_node_4',
            'patient_080_node_1', 'patient_088_node_1'
        ]

        self.data = {
            'patches': [],
            'tumor_ratio': [],
        }

        file_path = self.data_dir + f'center{self.center}_level2.hdf5'
        assert os.path.isfile(file_path)

        with h5py.File(file_path, 'r') as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                if self.phase == 'train':
                    if gname not in self.val_slides:
                        self.data['patches'].append(
                            torch.from_numpy(group['patches'][:]))
                        self.data['tumor_ratio'].append(
                            torch.from_numpy(group['tumor_ratio'][:]))
                else:
                    if gname in self.val_slides:
                        self.data['patches'].append(
                            torch.from_numpy(group['patches'][:]))
                        self.data['tumor_ratio'].append(
                            torch.from_numpy(group['tumor_ratio'][:]))

        self.data['patches'] = torch.cat(
            self.data['patches']).permute(0, 3, 1, 2)
        self.data['tumor_ratio'] = torch.cat(
            self.data['tumor_ratio']).unsqueeze(-1)

    def __getitem__(self, index):
        # get data
        x = self.data['patches'][index]  # data is stored in ByteTensors

        if self.norm:
            # output np.ndarray of dtype('uint8')
            x = self.norm.transform(x.permute(1, 2, 0).numpy())
            x = torch.from_numpy(x).permute(2, 0, 1)
        x = x.float().div_(255.)
        if self.transform:
            x = self.transform(x)

        # get label
        y = self.data['tumor_ratio'][index]
        # set labels larger than 0.01 to 1, i.e. tumor positive
        y = (y >= 0.01).float()
        return x, y

    def __len__(self):
        return len(self.data['patches'])


class MultipleCentersSeq(data.Dataset):
    """Adapted from https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
    Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(self, data_dir, center_indices, transform=transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), norm=None):
        super().__init__()
        self.data_dir = data_dir
        for i in center_indices:
            assert i in range(
                5), 'center_index is not valid. should be in range(5)'
        self.center_indices = center_indices
        self.transform = transform
        self.norm = norm

        self.file_paths = [self.data_dir +
                           f'center{c}_level2.hdf5' for c in self.center_indices]
        for f in self.file_paths:
            assert os.path.isfile(f)

        self.slide_names = []
        self.slide_lengths = []

        for f in self.file_paths:
            with h5py.File(f, 'r') as h5_file:
                for gname, group in h5_file.items():
                    self.slide_names.append((f, gname))
                    self.slide_lengths.append(
                        len(h5_file[gname]['tumor_ratio']))

        self.slide_indices = [sum(self.slide_lengths[:i+1])
                              for i in range(len(self.center_indices)*10-1)]

        self.data = {
            'patches': [],
            'tumor_ratio': [],
        }

        self.load_data(*self.slide_names[0])
        self.shift = 0

    def __getitem__(self, index):
        if index in self.slide_indices:
            slide = self.slide_indices.index(index)
            self.load_data(*self.slide_names[slide+1])
            self.shift = self.slide_indices[slide]
        # get data
        # data is stored in ByteTensors
        x = self.data['patches'][index-self.shift]

        if self.norm:
            # output np.ndarray of dtype('uint8')
            x = self.norm.transform(x.permute(1, 2, 0).numpy())
            x = torch.from_numpy(x).permute(2, 0, 1)

        x = x.float() / 255.

        if self.transform:
            x = self.transform(x)

        # get label
        y = self.data['tumor_ratio'][index-self.shift]
        # set labels larger than 0 to 1, i.e. tumor positive
        y = (y >= 0.01).float()
        return x, y

    def __len__(self):
        return sum(self.slide_lengths)

    def load_data(self, f, slide_name):
        self.data = {
            'patches': [],
            'tumor_ratio': [],
        }

        with h5py.File(f, 'r') as h5_file:
            for dname, ds in h5_file[slide_name].items():
                if dname == 'coordinates':
                    continue
                else:
                    self.data[dname].append(torch.from_numpy(ds[:]))

        self.data['patches'] = torch.cat(self.data['patches']).permute(0, 3, 1, 2)
        self.data['tumor_ratio'] = torch.cat(self.data['tumor_ratio']).unsqueeze(-1)


class TestCenterDataset(data.Dataset):
    """Adapted from https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
    Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(self, data_dir, center_index, transform=transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])):
        super().__init__()
        assert center_index in range(5), 'center_index is not valid. should be in range(5)'
        self.data_dir = data_dir
        self.center_index = center_index
        self.transform = transform

        self.file_path = self.data_dir + \
            f'center{self.center_index}_level2.hdf5'
        assert os.path.isfile(self.file_path)

        self.slide_names = []
        self.slide_lengths = []

        with h5py.File(self.file_path, 'r') as h5_file:
            for gname, group in h5_file.items():
                self.slide_names.append(gname)
                self.slide_lengths.append(len(h5_file[gname]['tumor_ratio']))

        self.slide_indices = [sum(self.slide_lengths[:i+1]) for i in range(9)]

        self.data = {
            'patches': [],
            'tumor_ratio': [],
        }

        self.load_slide(self.slide_names[0])
        self.shift = 0

    def __getitem__(self, index):
        if index in self.slide_indices:
            slide = self.slide_indices.index(index)
            self.load_slide(self.slide_names[slide+1])
            self.shift = self.slide_indices[slide]
        # get data
        # data is stored in ByteTensors
        x = self.data['patches'][index-self.shift]
        x = x.float() / 255.
        if self.transform:
            x = self.transform(x)

        # get label
        y = self.data['tumor_ratio'][index-self.shift]
        # set labels larger than 0 to 1, i.e. tumor positive
        y = (y >= 0.01).float()
        return x, y

    def __len__(self):
        return sum(self.slide_lengths)

    def load_slide(self, slide_name):
        self.data = {
            'patches': [],
            'tumor_ratio': [],
        }

        with h5py.File(self.file_path, 'r') as h5_file:
            for dname, ds in h5_file[slide_name].items():
                if dname == 'coordinates':
                    continue
                else:
                    self.data[dname].append(torch.from_numpy(ds[:]))

        self.data['patches'] = torch.cat(self.data['patches']).permute(0, 3, 1, 2)
        self.data['tumor_ratio'] = torch.cat(self.data['tumor_ratio']).unsqueeze(-1)
