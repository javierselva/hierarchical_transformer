# Original code from https://github.com/moliusimon/frnn

import numpy as np
import cv2
import h5py
import os
from sys import stdout as out
from random import shuffle
import subprocess
from data_utils import set_width_height, crop_video, load_categories_UCF101, classes, prepare_data_as_frames


# SCRIPT FOR PREPROCESSING THE UCF-101 DATASET
#   This script prepares the training and test partitions for the UCF-101 dataset. It will take the raw dataset from the
#   directory specified in DATA_PATH and place the processed partitions in SAVE_PATH.

DATA_PATH = '/data/ucf101/UCF-101/'
LABEL_PATH = '/data/ucf101/ucfTrainTestlist/'
SAVE_PATH = '/data/ucf101/frames'


# (height, width) indicate desired frame size. If only one provided the other is computed maintainig aspect ratio
# if none is provided, shortest_side is used as the desired frame size for the shortest side and the other is
# computed maintaining the aspect ratio
def prepare_sequence(file_path, height=-1, width=-1, shortest_side=-1, central_crop=None):
    # Read sequence frame by frame
    vidcap = cv2.VideoCapture(file_path)

    frames, (success, image) = [], vidcap.read()
    height, width = set_width_height(height, width, shortest_side, image.shape[0], image.shape[1])

    while success:
        out.write('\r Loading frame {} from {}'.format(len(frames), file_path))
        out.flush()
        # display_frame(image)
        image = cv2.resize(image[..., [2, 1, 0]], dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        # display_frame(image)

        frames.append(image)
        success, image = vidcap.read()

    vidcap.release()

    out_frames = np.stack(frames, axis=0)

    if central_crop is not None:
        out_frames = crop_video(out_frames, central_crop[0], central_crop[1])

    return out_frames, classes.get(file_path.split('/')[-2], 'error')


def prepare_partition(save_path, directory, partition,
                      height=-1, width=-1, shortest_side=-1, central_crop=None):
    # List video files in directory, prepare HDF5 database
    # Added class folders to the next line
    files = [os.path.join(directory, f.split('_')[1], f) for f in partition]
    database = h5py.File(save_path, 'w')

    # Create a dataset for each video file
    for i, f in enumerate(files):
        out.write('\r Processing sequence {}'.format(f))
        out.flush()
        sequence, label = prepare_sequence(f, height, width, shortest_side, central_crop=central_crop)
        # TODO Check compression/chunking for better efficiency; is storing each sequence as a dataset the best?
        #   Given that I won't load all frames from a sequence, maybe a chunk can be 3 frames
        dataset = database.create_dataset(str(i)+'_'+str(label), sequence.shape, dtype='u1')
        dataset[...] = sequence

    database.close()


# Loads specific partition from standard splits
def load_partition(file_path=LABEL_PATH, partition=1, split='train'):
    train_file = split + "list0" + str(partition) + ".txt"
    part_videos = list()
    with open(os.path.join(file_path, train_file), 'r') as f:
        text = f.read().split('\n')
        for line in text:
            if line:
                # For some reason, the label is only present in train lists
                if split == 'train':
                    file, _ = line.split(' ')
                else:
                    file = line
                part_videos.append(file.split('/')[-1])
    return part_videos


# path:str          indicates root directory for all categories
# partition:int     indicates which partition to use. 0 for random split of the classes
# train_size:float  if partition==0, it is used for train/test size in %
def partition_data(path, partition=0, train_size=.75):
    if not partition:
        # List files and group according to cateogry
        files, categories = [f for _, _, files in os.walk(path) for f in files if f.endswith('.avi')], {}
        for f in files:
            categories.setdefault(f.split('_')[1], []).append(f)
        # Partition each category using a 75%/25% split
        train, test = [], []
        for k in categories.keys():
            data = categories[k]
            shuffle(data)
            pivot = int(train_size * len(data))
            train.extend(data[:pivot])
            test.extend(data[pivot:])
    else:
        train = load_partition(partition=partition, split="train")
        test = load_partition(partition=partition, split="test")
    return train, test

def compute_class_balance(train,test):
    train_count = dict()
    test_count = dict()
    for vid in train:
        clas = vid.split('_')[1]
        train_count[clas] = train_count.setdefault(clas, 0) + 1
    for vid in test:
        clas = vid.split('_')[1]
        test_count[clas] = test_count.setdefault(clas, 0) + 1

    return train_count, test_count


if __name__ == '__main__':
    #### PREPARE CSV FILE
    # path = "/data-net/datasets/UCF101/ucfTrainTestlist/"
    # root_data_path = "/data-net/datasets/UCF101/UCF-101"
    # out_dir_base = "/data-net/datasets/UCF101/frames"
    # csv_file = "/data-net/datasets/UCF101/frames/annotations/train.csv"
    path = "/data/ucf101/ucfTrainTestlist/"
    root_data_path = "/data/ucf101/UCF-101"
    out_dir_base = "/data/ucf101/frames/videos"
    csv_file = "/data/ucf101/frames/annotations/train.csv"
    prepare_data_as_frames('ucf101', 'train', path, root_data_path, out_dir_base,
                           csv_file, 60, 80, partition=1)
    # csv_file = "/data-net/datasets/UCF101/frames/annotations/test.csv"
    # prepare_data_as_frames('ucf101', 'test', path, root_data_path, out_dir_base,
    #                        csv_file, -1, -1, partition=1)

    exit()

    train_file = '/train/' # 'ucf101_train_official_320.hdf5'
    test_file = '/test/' # 'ucf101_test_official_320.hdf5'

    load_categories_UCF101(LABEL_PATH)
    print('categories loaded')
    # Partition data
    # Had a problem with "HandstandPushups" as it did not match the name of files "HandStandPushups"
    #   Renamed the folder to be uppercase!! And also the reference in classInd.txt
    print('partitioning data ...')
    p_train, p_test = partition_data(DATA_PATH, partition=1)

    # tr1, te1 = compute_class_balance(p_train,p_test)
    #
    # p_train, p_test = partition_data(DATA_PATH, partition=0, train_size=.72)
    # tr2, te2 = compute_class_balance(p_train,p_test)
    # diff = 0
    # for cat in sorted(classes.keys()):
    #     diff += tr1[cat] - tr2[cat]
    #     print(cat, tr1[cat] - tr2[cat], tr1[cat], te1[cat], tr2[cat], te2[cat])
    # print("avg diff", diff / len(classes))
    print('preparing train partition ...')

    if os.path.exists(SAVE_PATH + train_file):
        print("Train partition seems to already exist at " + SAVE_PATH + train_file)
    else:
        # Prepare train partition
        prepare_partition(
            save_path=SAVE_PATH + train_file,
            directory=DATA_PATH,
            partition=p_train,
            height=60,
            width=80,                # If central_crop, these are used for rescaling first
            central_crop=None,    # None for no central_crop; (desired_w, desired_h) otherwise (only works in HDF5)
            format='fames'
        )

    print('preparing test partition ...')

    if os.path.exists(SAVE_PATH + test_file):
        print("Test partition seems to already exist at " + SAVE_PATH + test_file)
    else:
        # Prepare test partition
        prepare_partition(
            save_path=SAVE_PATH + test_file,
            directory=DATA_PATH,
            partition=p_test,
            height=60,
            width=80,
            central_crop=None,
            format='frames'
        )
    print("process finished")
