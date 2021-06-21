
import h5py
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torchvision

from torch.utils import data
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from tqdm.notebook import tqdm

from mdmm.model import MD_multi


class Args:
    concat = 1
    crop_size = 216 # only used as an argument for training
    dis_norm = None
    dis_scale = 3
    dis_spectral_norm = False
    dataroot ='data'
    gpu = 1
    input_dim = 3
    isDcontent = False
    nThreads = 4
    num_domains = 5
    nz = 8
    resume = '/home/haicu/sophia.wagner/projects/stain_color/stain_aug/mdmm_model.pth'


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
        assert phase in ['train', 'val'], 'phase is not valid. should be either train or val'
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
                        
        
        self.data['patches'] = torch.cat(self.data['patches']).permute(0, 3, 1, 2)
        self.data['tumor_ratio'] = torch.cat(self.data['tumor_ratio']).unsqueeze(-1)
                    
    def __getitem__(self, index): # ok
        # get data
        x = self.data['patches'][index] # data is stored in ByteTensors
        x = x.float().div_(255.)
        if self.transform:
            x = self.transform(x)

        # get label
        y = self.data['tumor_ratio'][index]
        y = (y >= 0.01).float() # set labels larger than 0 to 1, i.e. tumor positive
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
    def __init__(self, data_dir, center, phase, transform=transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), norm=None): # ok
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
                        self.data['patches'].append(torch.from_numpy(group['patches'][:]))
                        self.data['tumor_ratio'].append(torch.from_numpy(group['tumor_ratio'][:]))
                else:
                    if gname in self.val_slides:
                        self.data['patches'].append(torch.from_numpy(group['patches'][:]))
                        self.data['tumor_ratio'].append(torch.from_numpy(group['tumor_ratio'][:]))   
        
        self.data['patches'] = torch.cat(self.data['patches']).permute(0, 3, 1, 2)
        self.data['tumor_ratio'] = torch.cat(self.data['tumor_ratio']).unsqueeze(-1)
                    
    def __getitem__(self, index):
        # get data
        x = self.data['patches'][index] # data is stored in ByteTensors
        
        if self.norm:
            x = self.norm.transform(x.permute(1, 2, 0).numpy()) # output np.ndarray of dtype('uint8')
            x = torch.from_numpy(x).permute(2, 0, 1)
        x = x.float().div_(255.)
        if self.transform:
            x = self.transform(x)

        # get label
        y = self.data['tumor_ratio'][index]
        y = (y >= 0.01).float() # set labels larger than 0.01 to 1, i.e. tumor positive
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
    def __init__(self, data_dir, center_indices, transform=transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), norm=None): # ok
        super().__init__()
        self.data_dir = data_dir
        for i in center_indices:
            assert i in range(5), 'center_index is not valid. should be in range(5)'
        self.center_indices = center_indices
        self.transform = transform
        self.norm = norm
        
        self.file_paths = [self.data_dir + f'center{c}_level2.hdf5' for c in self.center_indices]
        for f in self.file_paths:
            assert os.path.isfile(f)
        
        self.slide_names = []
        self.slide_lengths = []
        
        for f in self.file_paths:
            with h5py.File(f, 'r') as h5_file:
                for gname, group in h5_file.items():
                    self.slide_names.append((f, gname))
                    self.slide_lengths.append(len(h5_file[gname]['tumor_ratio']))
        
        self.slide_indices = [sum(self.slide_lengths[:i+1])  for i in range(len(self.center_indices)*10-1)]
        
        self.data = {
            'patches': [],
            'tumor_ratio': [],
        }

#         if self.center_indices == [0,]:
#         self.transform = transforms.Normalize([0.6602, 0.5179, 0.6324], [0.2099, 0.2320, 0.1802])
#         elif self.center_indices == [1,]:
#             self.transform = transforms.Normalize([0.5504, 0.3831, 0.5325], [0.1837, 0.1977, 0.1605])
#         elif self.center_indices == [2,]:
#             self.transform = transforms.Normalize([0.4986, 0.3030, 0.6409], [0.2276, 0.1846, 0.1415])
#         elif self.center_indices == [3,]:
#             self.transform = transforms.Normalize([0.5671, 0.3789, 0.5278], [0.2191, 0.2049, 0.1658])
#         elif self.center_indices == [4,]:
#             self.transform = transforms.Normalize([0.7167, 0.5669, 0.7729], [0.1455, 0.1462, 0.0890])
#         elif self.center_indices == [1, 2, 3, 4]:
#             self.transform = transforms.Normalize([0.5784, 0.4018, 0.5978], [0.2117, 0.2076, 0.1759])
#         elif self.center_indices == [0, 2, 3, 4]:
#             self.transform = transforms.Normalize([0.6061, 0.4352, 0.6247], [0.2201, 0.2211, 0.1765])
#         elif self.center_indices == [0, 1, 3, 4]:
#             self.transform = transforms.Normalize([0.6111, 0.4457, 0.5976], [0.2065, 0.2155, 0.1822])
#         elif self.center_indices == [0, 1, 2, 4]:
#             self.transform = transforms.Normalize([0.6041, 0.4410, 0.6341], [0.2108, 0.2194, 0.1726])
#         elif self.center_indices == [0, 1, 2, 3]:
#             self.transform = transforms.Normalize([0.5709, 0.3971, 0.5730], [0.2167, 0.2178, 0.1718])
    
#         if self.mdmm_norm in range(5):
#             opts = Args()
#             aug_model = MD_multi(opts)
#             aug_model.resume(opts.resume, train=False)
#             aug_model.eval();
#             self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#             self.enc = aug_model.enc_c.to(self.device)
#             self.gen = aug_model.gen.to(self.device)

#             self.mean_domains = torch.Tensor([
#                 [ 0.3020, -2.6476, -0.9849, -0.7820, -0.2746,  0.3361,  0.1694, -1.2148],
#                 [ 0.1453, -1.2400, -0.9484,  0.9697, -2.0775,  0.7676, -0.5224, -0.2945],
#                 [ 2.1067, -1.8572,  0.0055,  1.2214, -2.9363,  2.0249, -0.4593, -0.9771],
#                 [ 0.8378, -2.1174, -0.6531,  0.2986, -1.3629, -0.1237, -0.3486, -1.0716],
#                 [ 1.6073,  1.9633, -0.3130, -1.9242, -0.9673,  2.4990, -2.2023, -1.4109],
#             ])

#             self.std_domains = torch.Tensor([
#                 [0.6550, 1.5427, 0.5444, 0.7254, 0.6701, 1.0214, 0.6245, 0.6886],
#                 [0.4143, 0.6543, 0.5891, 0.4592, 0.8944, 0.7046, 0.4441, 0.3668],
#                 [0.5576, 0.7634, 0.7875, 0.5220, 0.7943, 0.8918, 0.6000, 0.5018],
#                 [0.4157, 0.4104, 0.5158, 0.3498, 0.2365, 0.3612, 0.3375, 0.4214],
#                 [0.6154, 0.3440, 0.7032, 0.6220, 0.4496, 0.6488, 0.4886, 0.2989],
#             ])
#             print('was here')
#             print(self.device)

        self.load_data(*self.slide_names[0])
        self.shift = 0
                    
    def __getitem__(self, index):
        if index in self.slide_indices:
            slide = self.slide_indices.index(index)
            self.load_data(*self.slide_names[slide+1])
            self.shift = self.slide_indices[slide]
        # get data
        x = self.data['patches'][index-self.shift] # data is stored in ByteTensors
        
        if self.norm:
            x = self.norm.transform(x.permute(1, 2, 0).numpy()) # output np.ndarray of dtype('uint8')
            x = torch.from_numpy(x).permute(2, 0, 1)
            
        x = x.float() / 255.
            
        if self.transform:
            x = self.transform(x)
        
#         if self.mdmm_norm:
#             z_attr = (torch.randn((1, 8, )) * self.std_domains[self.mdmm_norm] + self.mean_domains[self.mdmm_norm]).to(self.device)
#             domain = torch.eye(5)[self.mdmm_norm].to(self.device).unsqueeze(0)
#             z_content = self.enc(x.unsqueeze().to(self.device))
#             x = self.gen(z_content, z_attr, domain).detach().squeeze(0).cpu() # in range [-1, 1]

        # get label
        y = self.data['tumor_ratio'][index-self.shift]
        y = (y >= 0.01).float() # set labels larger than 0 to 1, i.e. tumor positive
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
    def __init__(self, data_dir, center_index, transform=transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])): # ok
        super().__init__()
        assert center_index in range(5), 'center_index is not valid. should be in range(5)'
        self.data_dir = data_dir
        self.center_index = center_index
        self.transform = transform
        
        self.file_path = self.data_dir + f'center{self.center_index}_level2.hdf5'
        assert os.path.isfile(self.file_path)
        
        self.slide_names = []
        self.slide_lengths = []
        
        with h5py.File(self.file_path, 'r') as h5_file:
            for gname, group in h5_file.items():
                self.slide_names.append(gname)
                self.slide_lengths.append(len(h5_file[gname]['tumor_ratio']))
        
        self.slide_indices = [sum(self.slide_lengths[:i+1])  for i in range(9)]
        
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
        x = self.data['patches'][index-self.shift] # data is stored in ByteTensors
        x = x.float() / 255.
        if self.transform:
            x = self.transform(x)

        # get label
        y = self.data['tumor_ratio'][index-self.shift]
        y = (y >= 0.01).float() # set labels larger than 0 to 1, i.e. tumor positive
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


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
#         label_to_count = {}
#         for idx in tqdm(self.indices):
#             label = self._get_label(dataset, idx)
#             if label in label_to_count:
#                 label_to_count[label] += 1
#             else:
#                 label_to_count[label] = 1
#         print('labels', label_to_count)
                
#         # weight for each sample
#         weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
#                    for idx in self.indices]
#         self.weights = torch.DoubleTensor(weights)
        # wweight computed for training set of Center0Dataset
        self.weights = torch.DoubleTensor([5.1174e-05, 6.6357e-04])
        
    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset[idx][1].item()
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

    
def calculate_stats(dataloader, length):
    """calculate mean and standard deviation of the RGB channels of the given dataset in dataloader

    dataloader: iterates over dataset giving x (image), y (labels)
    length: number of samples in the dataset
    """
    total_sum = torch.zeros(3)
    sum_of_squared_error = torch.zeros(3)
    num_pixels = length * 512 * 512 # over all channels in first version

    # first calculate the mean
    for img, label in tqdm(dataloader):
        img = img.float() # now values are in range [0,1] for Center0Dataset
        
        total_sum += img.sum(dim=(0, 2, 3))
        
    mean = total_sum / num_pixels
    
    # standard deviation
    for img, label in tqdm(dataloader):
        img = img.float() # now valuse are in range [0,1] for Center0Dataset
        bs, _, _, _ = img.shape
        
        mean_dims = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        sum_of_squared_error += ((img - mean_dims).pow(2)).sum(dim=(0, 2, 3))
        
    std = torch.sqrt(sum_of_squared_error / num_pixels)
    
    return mean, std


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(10, 10))
#     cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names, rotation=90)
    
    # Normalize the confusion matrix.
#     cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size sum(n_samples)
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = self.labels.unique().tolist()
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = sum(self.n_samples)

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            indices = []
            for class_ in self.labels_set:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples[int(class_)]])
                self.used_label_indices_count[class_] += self.n_samples[int(class_)]
                if self.used_label_indices_count[class_] + self.n_samples[int(class_)] > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            np.random.shuffle(indices)
            yield indices
            self.count += sum(self.n_samples)

    def __len__(self):
        return self.n_dataset // self.batch_size