from torch.utils.data import Dataset
from torch import tensor
from torchvision.transforms import RandomCrop, CenterCrop
import numpy as np
from math import ceil
from os.path import join as pjoin
from data_utils.data_utils import read_csv_file
from decord import VideoReader, cpu

# expected raw video data structure:
#  dataset_home_path/
#       |_annotations/
#       |    |_train.csv
#       |    |_test.csv
#       |    |_val.csv
#       |_videos
#            |_train
#            |    |_vid_1.mp4
#            |    |_...
#            |_test
#            |    |_vid_1.mp4
#            |    |_...
#            |_val
#                 |_vid_1.mp4
#                 |_...

class CustomDataset(Dataset):
    def __init__(self, config, mode, dataset='ucf101', transform=None):
        self.params = config['data'][dataset]['sampler']
        self.data_root = config['data'][dataset]['location']['SAVE_PATH']
        self.videos_path = pjoin(self.data_root,'videos')
        self.s_len = self.params['length']
        self.s_freq = self.params['frequency']
        self.s_window = (self.s_len - 1) * self.s_freq + 1
        self.mode = mode
        self.return_idx = self.mode == 'test'
        self.num_classes = config['data'][dataset]['num_classes']
        self.transform = transform

        self.data, self.labels, self.lengths, self.resolutions = read_csv_file(
                                                                 pjoin(self.data_root, 'annotations', f'{mode}.csv'))
        self.mapping, self.offset = list(), list()

        assert all([0 <= x < self.num_classes for x in self.labels]), "Labels must be between 0 and " + str(self.num_classes - 1)

        # Remove videos with sequence length smaller than the sample window
        self.data, self.lengths, self.labels, self.resolutions = zip(
            *[(v, l, b, r) for v, l, b, r in zip(self.data, self.lengths, self.labels, self.resolutions)
              if l >= self.s_window])

        step = self.params['step']
        # Prepare mapping (sample to sequence)
        self.mapping = np.concatenate(tuple(
            np.array([i] * ceil((l - self.s_window + 1) / step)) for i, l in enumerate(self.lengths)
        ), axis=0)

        # Prepare offset (sample within sequence)
        self.offset = np.concatenate(tuple(
            np.arange(0, l - self.s_window + 1, step) for i, l in enumerate(self.lengths)
        ), axis=0)

        assert len(self.mapping) == len(self.offset)

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
        off = self.offset[idx]
        path = self.data[self.mapping[idx]]
        reso = self.resolutions[self.mapping[idx]]
        label = self.labels[self.mapping[idx]]
        # Load frames
        seq = self.load_video_frames(path, off, reso)
        # TODO maybe redo this casting/transform thing, did it in a rush, may not be the most optimal
        seq = tensor(np.cast[np.float32](seq))
        if self.transform:
            # As most transforms assume N x C x H x W, permutting by default
            seq = self.transform(seq.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.return_idx:
            return seq / 255, label, self.mapping[idx]
        else:
            return seq / 255, label

    def load_video_frames(self, path, off, res):
        # Rescale so the shortest side is 256
        if res[0] < res[1]:
            h, w = 256, int(256 * res[1] / res[0])
        else:
            h, w = int(256 * res[0] / res[1]), 256
        vr = VideoReader(pjoin(self.videos_path, path), ctx=cpu(0), width=w, height=h)
        seq = vr.get_batch(list(range(off, off + self.s_window, self.s_freq))).asnumpy()
        return seq

    def get_num_batches(self, batch_size):
        return self.num_samples/batch_size
