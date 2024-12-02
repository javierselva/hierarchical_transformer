from preprocess_ucf101 import prepare_sequence
from data_utils.data_utils import load_files_and_id2label_ssv2, prepare_data_as_frames, create_csv_for_raw_videos
import h5py

DATA_PATH = '/data-net/datasets/SSv2/20bn-something-something-v2'
SAVE_PATH = '/data-net/datasets/SSv2/features/'
LABEL_PATH = '/data-net/datasets/SSv2/20bn-something-something-labels/'

def prepare_partition(save_path, data_path, label_path, partition,
                      height=-1, width=-1, shortest_side=-1, central_crop=None, format='hdf5'):

    files, id_to_label = load_files_and_id2label_ssv2(label_path, partition, data_path)

    ### CREATE HDF5 FILE
    database = h5py.File(save_path, 'w')

    # Create a dataset for each video file
    for i, f in enumerate(files):
        sequence, _ = prepare_sequence(f, height, width, shortest_side, central_crop)
        label = id_to_label[f.split('.')[-2].split('/')[-1]]
        dataset = database.create_dataset(str(i)+'_'+str(label), sequence.shape, dtype='u1')
        dataset[...] = sequence

    database.close()

if __name__ == '__main__':
    #### PREPARE CSV FILE
    path = "/data-net/datasets/SSv2/20bn-something-something-labels/"
    root_data_path = "/data-net/datasets/SSv2/20bn-something-something-v2/"
    #csv_file = "/data-net/datasets/SSv2/frames/annotations/train.csv"
    out_dir_base = "/data-local/data1-ssd/jselva/ssv2/raw_videos/annotations"
    create_csv_for_raw_videos(root_data_path, path, out_dir_base,
                              'train', 'ssv2', fieldnames=['path', 'label'])
    #csv_file = "/data-net/datasets/SSv2/frames/annotations/test.csv"
    create_csv_for_raw_videos(root_data_path, path, out_dir_base,
                              'test', 'ssv2', fieldnames=['path', 'label'])
    #csv_file = "/data-net/datasets/SSv2/frames/annotations/val.csv"
    create_csv_for_raw_videos(root_data_path, path, out_dir_base,
                              'val', 'ssv2', fieldnames=['path', 'label'])


    exit()
    train_file = 'ssv2_train_256.hdf5'
    val_file = 'ssv2_val_256.hdf5'
    test_file = 'ssv2_test_256.hdf5'

    print('preparing train partition ...')

    if os.path.exists(SAVE_PATH + train_file):
        print("Train partition seems to already exist at " + SAVE_PATH + train_file)
    else:
        # Prepare train partition
        prepare_partition(
            save_path=SAVE_PATH + train_file,
            data_path=DATA_PATH,
            label_path=LABEL_PATH,
            partition='train',
            shortest_side=256
        )

    print('preparing validation partition ...')

    if os.path.exists(SAVE_PATH + val_file):
        print("Val partition seems to already exist at " + SAVE_PATH + val_file)
    else:
        # Prepare test partition
        prepare_partition(
            save_path=SAVE_PATH + val_file,
            data_path=DATA_PATH,
            label_path=LABEL_PATH,
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
            data_path=DATA_PATH,
            label_path=LABEL_PATH,
            partition='test',
            shortest_side=256
        )

    print("process finished")