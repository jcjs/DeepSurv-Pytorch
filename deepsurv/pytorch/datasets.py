"""
Pytorch datasets for Toy Model
"""
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader


def read_data(filename):
    """
    Reads hdf5 simulated data
    """
    groups = ['test', 'train', 'valid', 'viz']
    output_dict = {}
    f = h5py.File(filename, 'r')
    for group in groups:
        try:
            e = f[group]['e'].value
            t = f[group]['t'].value
            x = f[group]['x'].value

            sort_idx = np.argsort(t)[::-1]
            x = x[sort_idx]
            e = e[sort_idx]
            t = t[sort_idx]

            output_dict[group] = [e, t, x]
        except Exception as e:
            continue

    f.close()
    return output_dict


class SurvivalDataset(Dataset):
    """
    Class to work with survival data
    """
    def __init__(self, e, t, x):
        assert len(x) == len(t)
        assert len(x) == len(e)

        self.e = e.astype(np.float32)
        self.t = t
        self.x = x
        self.len = len(e)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        Returns e, t, x
        """
        return self.e[idx], self.t[idx], self.x[idx]

def get_datasets(filename):
    """
    Returns pytorch datasets
    """
    raw_datasets = read_data(filename)
    out = {}
    for d_type in raw_datasets.keys():
        out[d_type] = SurvivalDataset(*raw_datasets[d_type])
    return out

def get_loaders(datasets, b_size=10):
    """
    Returns data loaders
    """
    out = {}
    for d_type in datasets.keys():
        out[d_type] = DataLoader(datasets[d_type], batch_size=b_size, shuffle=True, drop_last=True)
    return out