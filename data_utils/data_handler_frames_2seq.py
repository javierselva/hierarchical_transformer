from torch.utils.data import Dataset
from torch import tensor, stack
from torchvision.transforms import RandomCrop, CenterCrop
from torchvision.io import read_image
from data_utils.data_handler_frames import prepend_zeros
from data_utils.data_utils import read_csv_file
import numpy as np
from math import ceil
from os.path import join as pjoin


# Two main ways to do this:
# 1. Read all frames into memory and then sample from them the various sequences (dataset len should be defined by pairs of sequences)
# 2. Leave it as is. Sample the sequences on the fly by selecting two random sequences from a given clip. (dataset len is number of pairs of sequences)
def get_random_pairing(seq_range):
    if seq_range[0] == seq_range[1]-1:
        return [seq_range[0], seq_range[0]]
    pair1 = np.random.randint(*seq_range)
    pair2 = np.random.randint(*seq_range)
    while pair1 == pair2:
        pair2 = np.random.randint(*seq_range)
    return [pair1, pair2]

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
        self.return_single_sequence = self.return_idx
        self.num_classes = config['data'][dataset]['num_classes']
        self.transform = transform

        self.data, self.labels, self.lengths, _ = read_csv_file(pjoin(self.data_root, 'annotations', f'{mode}.csv'))
        self.mapping, self.offset = list(), list()

        assert all([0 <= x < self.num_classes for x in self.labels]), "Labels must be between 0 and " + str(self.num_classes - 1)

        # Remove videos with sequence length smaller than the sample window
        self.data, self.lengths, self.labels = zip(
            *[(v, l, b) for v, l, b in zip(self.data, self.lengths, self.labels) if l >= self.s_window])

        step = self.params['step']
        # Prepare mapping (sample to sequence)
        self.mapping = np.concatenate(tuple(
            np.array([i] * ceil((l - self.s_window + 1) / step)) for i, l in enumerate(self.lengths)
        ), axis=0)

        # Prepare offset (within sample frame offset)
        self.offset = np.concatenate(tuple(
            np.arange(0, l - self.s_window + 1, step) for l in self.lengths), axis=0)

        assert len(self.mapping) == len(self.offset)

        # Store the range of sequences belonging to each video
        self.seq_id2range = dict()
        i = 0
        while i < len(self.mapping):
            j = i
            while j < len(self.mapping) and self.mapping[i] == self.mapping[j]:
                j += 1
            self.seq_id2range[self.mapping[i]] = (i, j)
            i = j

        self.num_samples = len(self.mapping)

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
        if self.return_single_sequence:  # For evaluation, just one sequence!
            frames = self.load_sequence(idx)
            if self.return_idx:
                return frames / 255, tensor(self.labels[self.mapping[idx]]), self.mapping[idx]
            else:
                return frames / 255, tensor(self.labels[self.mapping[idx]])
        else:  # For training, two sequences!
            seq_num = self.mapping[idx]
            seq_range = self.seq_id2range[seq_num]
            pair = get_random_pairing(seq_range)
            pair_of_sequences = list()
            for idx in pair:
                pair_of_sequences.append(self.load_sequence(idx) / 255)
            return pair_of_sequences[0], pair_of_sequences[1], tensor(self.labels[self.mapping[idx]])

    def load_sequence(self, idx):
        off = self.offset[idx]
        frames = list()
        for f in range(off, off + self.s_window, self.s_freq):
            frames.append(self.open_frame(self.data[self.mapping[idx]], f))
        frames = stack(frames)
        if self.transform:
            frames = self.transform(frames)
        return frames.permute(0, 2, 3, 1)

    # Given the path to a video and a frame index, load and return the frame
    def open_frame(self, video_path, frame_index):
        return read_image(pjoin(self.video_path,video_path, 'frame_' + prepend_zeros(frame_index+1) + '.jpg'))

    def get_num_batches(self, batch_size):
        return self.num_samples/batch_size

    def set_single_sequence(self):
        self.return_single_sequence = True

    def reset_single_sequence(self):
        self.return_single_sequence = self.return_idx

