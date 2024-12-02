from preprocess_ucf101 import prepare_sequence
import h5py
import os
import cv2
import sys
from data_utils import load_categories_K400, get_video_fps, load_k400_id2label, prepare_data_as_frames, create_csv_for_raw_videos

DATA_PATH = '/data-net/datasets/Kinetics-400/second_attempt/k400'
SAVE_PATH = '/data-net/datasets/Kinetics-400/features/'
LABEL_PATH = '/data-net/datasets/Kinetics-400/second_attempt/k400/annotations'

def print_all_faulty_videos(do_remove=False):
    for partition in ['train', 'test', 'val']:
        partition_path = os.path.join(DATA_PATH, partition)
        files = [os.path.join(partition_path, f) for f in os.listdir(os.path.join(partition_path))
                 if os.path.isfile(os.path.join(partition_path, f))]

        for file_path in files:
            vidcap = cv2.VideoCapture(file_path)

            _, (success, _) = [], vidcap.read()
            if not success:
                print(file_path)
                if do_remove:
                    os.remove(file_path)
            else:
                num_frames = 0
                while success:
                    num_frames += 1
                    (success, _) = vidcap.read()

                expected_frames = int(get_video_fps(file_path)*10)
                if num_frames < expected_frames:
                    print(str(num_frames) + '/' + str(expected_frames), file_path)
                    if do_remove:
                        os.remove(file_path)
            vidcap.release()

def prepare_partition(save_path, directory, partition,
                      height=-1, width=-1, shortest_side=-1, central_crop=None):
    partition_path = os.path.join(directory, partition)
    files = [os.path.join(partition_path, f) for f in os.listdir(partition_path)
                                                if os.path.isfile(os.path.join(partition_path, f))]

    id_to_label = load_k400_id2label(LABEL_PATH,partition)

    database = h5py.File(save_path, 'w')

    # Create a dataset for each video file
    for i, f in enumerate(files):
        try:
            sequence, _ = prepare_sequence(f, height, width, shortest_side, central_crop)
            label = id_to_label[f.split('/')[-1][:11]]
            dataset = database.create_dataset(str(i)+'_'+str(label), sequence.shape, dtype='u1')
            dataset[...] = sequence
        except:
            print("Failed to open ", f)

    database.close()

if __name__ == '__main__':
    ### PREPARE CSV FILE (videos)
    path = "/data-net/datasets/Kinetics-400/k400/annotations"
    root_data_path = "/data-net/datasets/Kinetics-400/k400"
    out_dir_base = "/data-net/datasets/Kinetics-400/raw_video_csvs"
    create_csv_for_raw_videos(root_data_path, path, out_dir_base, 'train', 'k400')
    create_csv_for_raw_videos(root_data_path, path, out_dir_base, 'test', 'k400')
    create_csv_for_raw_videos(root_data_path, path, out_dir_base, 'val', 'k400')

    exit()
    #### PREPARE CSV FILE (frames)
    path = "/data-net/datasets/Kinetics-400/annotations"
    root_data_path = "/data-net/datasets/Kinetics-400/k400"
    out_dir_base = "/data-net/datasets/Kinetics-400/frames"
    csv_file = "/data-net/datasets/Kinetics-400/frames/annotations/train.csv"
    prepare_data_as_frames('k400', 'train', path, root_data_path, out_dir_base,
                           csv_file, -1, -1)
    csv_file = "/data-net/datasets/Kinetics-400/frames/annotations/test.csv"
    prepare_data_as_frames('k400', 'test', path, root_data_path, out_dir_base,
                           csv_file, -1, -1)
    csv_file = "/data-net/datasets/Kinetics-400/frames/annotations/val.csv"
    prepare_data_as_frames('k400', 'val', path, root_data_path, out_dir_base,
                           csv_file, -1, -1)

    exit()

    train_file = 'k400_train_256.hdf5'
    val_file = 'k400_val_256.hdf5'
    test_file = 'k400_test_256.hdf5'

    load_categories_K400(DATA_PATH)
    print('categories loaded')

    print('preparing train partition ...')

    if os.path.exists(SAVE_PATH + train_file):
        print("Train partition seems to already exist at " + SAVE_PATH + train_file)
    else:
        # Prepare train partition
        prepare_partition(
            save_path=SAVE_PATH + train_file,
            directory=DATA_PATH,
            partition='train',
            shortest_side=256
        )

    print('preparing validation partition ...')

    if os.path.exists(SAVE_PATH + val_file):
        print("Test partition seems to already exist at " + SAVE_PATH + val_file)
    else:
        # Prepare test partition
        prepare_partition(
            save_path=SAVE_PATH + val_file,
            directory=DATA_PATH,
            partition='val',
            shortest_side=256
        )

    print('preparing test partition ...')

    if os.path.exists(SAVE_PATH + test_file):
        print("Test partition seems to already exist at " + SAVE_PATH + test_file)
    else:
        # Prepare test partition
        prepare_partition(
            save_path=SAVE_PATH + test_file,
            directory=DATA_PATH,
            partition='test',
            shortest_side=256
        )

    print("process finished")