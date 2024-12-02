from torch.utils.data import Dataset
from torch import tensor, stack
from torchvision.transforms import RandomCrop, CenterCrop
from torchvision.io import read_image
from data_utils.data_utils import read_csv_file
import numpy as np
from math import ceil
from os.path import join as pjoin

def prepend_zeros(i,desired_length=3):
    o = str(i)
    while len(o) < desired_length:
        o = '0' + o
    return o

# Given an integer x, sample y numbers from 0 to x-1 such that none of the y numbers are the same
def sample_without_repeats(x, y):
    if x == y:
        return list(range(x))
    else:
        sampling = np.random.choice(x, y, replace=False)
        # sort samples before returning
        sampling.sort()
        return sampling

#### EXPECTED TREE STRUCTURE TO BE READ
# data_path/videos:
#   |_ train or test or val
#   |  |_vid 1
#   |  |   |_ frame_000.jpg
#   |  |   |_ frame_001.jpg
#   |  |   |_ ...
#   |  |   |_ frame_xyz.jpg
#   |  |_...
#   |  |_vid N
#   |  |   |_ frame_000.jpg
#   |  |   |_ ...
#   |  |   |_ frame_xyz.jpg
# TODO some things done for this data loader have not been addressed for hdf5 or decord (e.g. max_seq_per_video)
class CustomDataset(Dataset):
    def __init__(self, config, mode, dataset='ucf101', transform=None):
        self.params = config['data'][dataset]['sampler']
        self.data_root = config['data'][dataset]['location']['SAVE_PATH']
        self.video_path = pjoin(self.data_root,'videos')
        self.s_len = self.params['length']
        self.s_freq = self.params['frequency']
        self.s_window = (self.s_len - 1) * self.s_freq + 1
        self.mode = mode
        self.return_idx = self.mode == 'test' or self.mode == 'val'
        self.num_classes = config['data'][dataset]['num_classes']
        self.transform = transform
        self.max_seq_per_video = config['data']['max_seq_per_video'] if self.mode == 'train' else 0

        self.data, self.labels, self.lengths, _ = read_csv_file(pjoin(self.data_root, 'annotations', f'{mode}.csv'))
        self.mapping, self.offset = list(), list()

        assert all([0 <= x < self.num_classes for x in self.labels]), "Labels must be between 0 and " + str(self.num_classes - 1)

        # Remove videos with sequence length smaller than the sample window
        self.data, self.lengths, self.labels = zip(
            *[(v, l, b) for v, l, b in zip(self.data, self.lengths, self.labels) if l >= self.s_window])

        self.compute_mapping_and_offset()

        if self.transform is None:
            crop_size = (self.params['height'], self.params['width'])
            if self.mode == 'train':
                self.transform = RandomCrop(crop_size)
            elif self.mode == 'val':
                self.transform = CenterCrop(crop_size)
            else:
                self.transform = CenterCrop(crop_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        off = self.offset[idx]
        frames = list()
        for f in range(off, off + self.s_window, self.s_freq):
            frames.append(self.open_frame(self.data[self.mapping[idx]], f))
        frames = stack(frames)
        if self.transform:
            frames = self.transform(frames)

        # TODO avoid this permute as it is reversed in the model input
        frames = frames.permute(0, 2, 3, 1)

        if self.return_idx:
            return frames / 255, tensor(self.labels[self.mapping[idx]]), self.mapping[idx]
        else:
            return frames / 255, tensor(self.labels[self.mapping[idx]])

    # Given the path to a video and a frame index, load and return the frame
    def open_frame(self, video_path, frame_index):
        return read_image(pjoin(self.video_path,video_path, 'frame_' + prepend_zeros(frame_index+1) + '.jpg'))

    def get_num_batches(self, batch_size):
        return self.num_samples/batch_size

    def compute_mapping_and_offset(self):
        step = self.params['step']
        # Prepare mapping (sample to sequence)
        self.mapping = np.concatenate(tuple(
            np.array([i] * ceil((l - self.s_window + 1) / step)) for i, l in enumerate(self.lengths)
        ), axis=0)

        # Prepare offset (within sample frame offset)
        self.offset = np.concatenate(tuple(
            np.arange(0, l - self.s_window + 1, step) for l in self.lengths), axis=0)

        new_mapping = list()
        new_offset = list()
        if self.max_seq_per_video > 0:
            i = 0
            while i < len(self.mapping):
                j = i + 1
                # Find all sequences for a given video
                while j < len(self.mapping) and self.mapping[j] == self.mapping[i]:
                    j += 1
                # If the number of sequences is greater than the maximum allowed
                if j - i > self.max_seq_per_video:
                    # Randomly sample the sequences
                    new_mapping.extend(self.mapping[i + k] for k in sample_without_repeats(j - i, self.max_seq_per_video))
                    new_offset.extend(self.offset[i + k] for k in sample_without_repeats(j - i, self.max_seq_per_video))
                else:
                    new_mapping.extend(self.mapping[i:j])
                    new_offset.extend(self.offset[i:j])
                i = j

            self.mapping = new_mapping
            self.offset = new_offset
        assert len(self.mapping) == len(self.offset)

        self.num_samples = len(self.mapping)

