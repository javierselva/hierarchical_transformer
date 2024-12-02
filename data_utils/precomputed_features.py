import h5py
from os import path, remove
from torch.utils.data import Dataset
from numpy import float32

class PreComputedFeatures(Dataset):
    def __init__(self, num_samples, feature_dim, save_path='/tmp', save_file=''):
        self.num_batches = 0
        self.num_samples = num_samples
        self.idx = 0

        self.data_path = path.join(save_path, save_file) + '.hdf5'

        self.database = h5py.File(self.data_path, 'w')
        self.data = self.database.create_dataset('data', (num_samples, feature_dim), dtype=float32, chunks=True)
        self.labels = self.database.create_dataset('labels', (num_samples), dtype='u1')

    def get_num_batches(self):
        return self.num_batches

    def __getitem__(self, idx):
        if self.database is None:
            self.load()
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples

    def add_batch(self, batch, labels):
        self.num_batches += 1

        self.data[self.idx:self.idx + batch.shape[0], :] = batch
        self.labels[self.idx:self.idx + labels.shape[0]] = labels

        self.idx += batch.shape[0]

    def load(self):
        self.database = h5py.File(self.data_path, 'r')
        self.data = self.database['data']
        self.labels = self.database['labels']

    def finished_saving(self):
        self.database.close()
        self.database = None

    def delete_file(self):
        remove(self.data_path)