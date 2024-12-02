from torch.utils.data import Dataset
from torch import tensor
import numpy as np
from math import ceil
import h5py
from os.path import join as pjoin


class CustomDataset(Dataset):
    # Different from the other datasets, this one does not do automatic CentralCrop, assumes HDF5 generated with desired size
    def __init__(self, config, mode, dataset='ucf101', transform=None):
        self.params = config['data'][dataset]['sampler']
        self.location = config['data'][dataset]['location']
        self.s_len = self.params['length']
        self.s_freq = self.params['frequency']
        self.s_window = (self.s_len - 1) * self.s_freq + 1
        self.w = self.params['width']
        self.h = self.params['height']
        self.mode = mode
        self.return_idx = self.mode == 'test'
        self.num_classes = config['data'][dataset]['num_classes']

        self.dataset, self.data, self.lengths, self.labels, self.mapping, self.offset = None, list(), list(), list(), list(), list()

        self.test_size_mismatch = True

        self.open_hdf5()

        self.dataset.close()
        del self.dataset
        self.dataset = None

        self.num_samples = len(self.mapping)
        self.transform = transform

    # So, apparently, hdf5 does not support multi-threading on a single instance.
    # What it does support is that multiple threads open the same file.
    # In that sense, following https://github.com/pytorch/pytorch/issues/11929
    #  the proposal is to open one instance on the first call to __getitem__
    def open_hdf5(self):
        # Prepare reader for each video sequence
        path = self.location['SAVE_PATH']
        self.dataset = h5py.File(pjoin(path, self.location[self.mode + '_file']), 'r')
        self.data, self.labels = zip(*[(v, int(k.split('_')[1]))
                                       for k, v in
                                       self.dataset.items()])
        self.lengths = [len(v) for v in self.data]

        assert all([0 <= x < self.num_classes for x in self.labels]), "Labels must be between 0 and " + str(self.num_classes - 1)

        # Remove videos with sequence length smaller than the sample window
        self.data, self.lengths, self.labels = zip(
            *[(v, l, b) for v, l, b in zip(self.data, self.lengths, self.labels) if l >= self.s_window])

        step = self.params['step']
        # Prepare mapping (sample to sequence)
        self.mapping = np.concatenate(tuple(
            np.array([i] * ceil((l - self.s_window + 1) / step)) for i, l in enumerate(self.lengths)
        ), axis=0)

        # Prepare padding (sample to sequence padding)
        self.offset = np.concatenate(tuple(
            np.arange(0, l - self.s_window + 1, step) for i, l in enumerate(self.lengths)
        ), axis=0)

        assert len(self.mapping) == len(self.offset)

        if self.test_size_mismatch:
            sample_shape = self.data[0].shape[1:3]
            if sample_shape[0] != self.h or sample_shape[1] != self.w:
                print("WARNING: Found mismatch between frame sizes in data_config.yaml (" + str((self.h, self.w)) +
                      ") and the ones in the provided .hdf5 file (" + str(sample_shape) +
                      "). Make sure this is desired behaviour (e.g., to have a smaller size after data augmentation)")
            self.test_size_mismatch = False

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.dataset is None:
            self.open_hdf5()
        off = self.offset[idx]
        seq = self.data[self.mapping[idx]][off:off + self.s_window:self.s_freq]
        label = self.labels[self.mapping[idx]]
        # TODO maybe redo this casting/transform thing, did it in a rush, may not be the most optimal
        seq = tensor(np.cast[np.float32](seq))
        if self.transform:
            # As most transforms asume N x C x H x W, permutting by default
            seq = self.transform(seq.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.return_idx:
            return seq / 255, label, self.mapping[idx]
        else:
            return seq / 255, label

    def get_num_batches(self, batch_size):
        return self.num_samples/batch_size


