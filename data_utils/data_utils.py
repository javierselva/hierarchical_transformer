import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
#import cv2

classes = dict()

# Given a dictionary representing a histogram, this function prints a histogram using ascii characters
def ascii_histogram_plot(hist,max_width=80):
    for k in sorted(hist, key=lambda x:int(x)):
        print(k, ': ', '*' * round(hist[k]/max(hist.values())*max_width))

# This function loads a csv file, loads all values of a given column, and returns a dictionary representing a histogram
def compute_histogram_of_csv_column(csv_path, column):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        hist = dict()
        for row in reader:
            val = row[column]
            if val in hist:
                hist[val] += 1
            else:
                hist[val] = 1
    return hist

def read_csv_file(file_path):
    path_list = list()
    label_list = list()
    num_frames_list = list()
    resolution_list = list()
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            path_list.append(row['path'])
            label_list.append(int(row['label']))
            num_frames_list.append(int(row['num_frames']))
            resolution_list.append((int(row['height']), int(row['width'])))
    return path_list, label_list, num_frames_list, resolution_list

def get_num_frames_ffprobe(video_path):
    result = subprocess.Popen(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", video_path], stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    num_frames = int(result.stdout.read())
    return num_frames

# Accurate frame count if using raw videos
def count_num_frames_opencv(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    num_frames = 0
    while ret:
        num_frames += 1
        ret, frame = cap.read()
    cap.release()
    return num_frames

# Counts number of frames saved as images in a given directory
def count_number_of_frames_as_images(path):
    return len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])

# Tree structure for raw videos:
# root_data_dir/
#   |_ {train or test or val}/
#   |    |_ video1.mp4 (names of video not necessarily with this format)
#   |    |_ video2.mp4
#   |    |_ ...
# Given that three strucuture, this function creates a csv file for each split (train, test, val)
# with the following structure: path, label, num_frames, height, width
def create_csv_for_raw_videos(data_root, label_path, out_dir, split, dataset,
                              fieldnames=None, partition=0):
    if dataset == 'ssv2':
        paths, labels = load_ssv2_paths_and_labels(label_path, split, data_root)
    elif dataset == 'k400':
        paths, labels = load_k400_paths_and_labels(label_path, split, data_root)
    elif dataset == 'ucf101':
        paths, labels = load_ucf_paths_and_labels(label_path, split, partition)
    else:
        raise ValueError('Dataset not supported')

    if fieldnames is None:
        fieldnames = ['path', 'label', 'num_frames', 'height', 'width']

    num_failed_videos = 0
    csv_path = os.path.join(out_dir, split + '.csv')
    with open(csv_path, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for p,l in zip(paths,labels):
            dict_to_write = dict()
            actual_video_path = os.path.join(data_root,p)
            dict_to_write['path'] = p
            dict_to_write['label'] = l
            try:
                if 'num_frames' in fieldnames:
                    dict_to_write['num_frames'] = count_num_frames_opencv(actual_video_path)
                if 'height' in fieldnames:
                    dict_to_write['height'], dict_to_write['width'] = get_video_resolution(actual_video_path)
            except:
                print('Error reading video: ', p)
                num_failed_videos += 1
                continue
            writer.writerow(dict_to_write)
    print('Number of failed videos: ', num_failed_videos)

# Extracts frames from the whole dataset and generates csv file
def prepare_data_as_frames(dataset, split, label_path, root_data_dir, out_dir_base,
                           csv_path, height, width, shortest_side=256, partition=0):
    if dataset == 'ssv2':
        paths, labels = load_ssv2_paths_and_labels(label_path, split, root_data_dir)
    elif dataset == 'k400':
        paths, labels = load_k400_paths_and_labels(label_path, split, root_data_dir)
    elif dataset == 'ucf101':
        paths, labels = load_ucf_paths_and_labels(label_path, split, partition)
    else:
        raise ValueError('Dataset not supported')

    num_failed_videos = 0

    open_mode = 'w'
    if os.path.exists(csv_path):
        print('Provided CSV already exists. Appending to existing csv file: ', csv_path)
        open_mode = 'a'

    with open(csv_path, mode=open_mode) as csv_file:
        fieldnames = ['path', 'label', 'num_frames', 'height', 'width']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if open_mode == 'w':
            writer.writeheader()
        for p,l in zip(paths,labels):
            actual_video_path = os.path.join(root_data_dir,p)
            h, w = get_video_resolution(actual_video_path)
            height, width = set_width_height(height, width, shortest_side, h, w)

            out_dir = os.path.join(out_dir_base, 'videos', split, p.split('/')[-1].split('.')[0])
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, )
            else:
                print('Frames already extracted for video: ', p)
                continue
            try:
                command = ["ffmpeg",
                           "-nostdin", "-i", actual_video_path,
                           "-vf", "scale=" + str(width) + ":" + str(height),
                           os.path.join(out_dir,"frame_%03d.jpg")]

                subprocess.run(command, capture_output=True)
                num_frames = count_number_of_frames_as_images(out_dir)
                assert num_frames > 1
            except:
                print('Error reading video: ', p)
                num_failed_videos += 1
                continue

            writer.writerow({'path': os.path.join(split, p.split('/')[-1].split('.')[0]),
                             'label': l,
                             'num_frames': num_frames,
                             'height': height,
                             'width': width})
    print('Number of failed videos: ', num_failed_videos)

# These three functions do not add the data_path to the paths.
# (data loader and frame extractors deal with different paths)
def load_ssv2_paths_and_labels(path,split,root_data_dir):
    all_files, id_to_label = load_files_and_id2label_ssv2(path,split,root_data_dir)
    paths = list()
    labels = list()

    for file in all_files:
        video_id = file.split('/')[-1].split('.')[0]
        paths.append(os.path.join(file.split('/')[-1]))
        labels.append(id_to_label[video_id])

    return paths, labels

# root_data_dir should end with k400
# Path should end with "k400/annotations/
def load_k400_paths_and_labels(path,split,root_data_dir):
    split_path = os.path.join(root_data_dir,split)
    load_categories_K400(path)
    id_to_label = load_k400_id2label(path, split)

    all_files = [f for f in os.listdir(split_path)
             if os.path.isfile(os.path.join(split_path, f))]

    paths = list()
    labels = list()

    for file in all_files:
        if file == '':
            continue
        video_id = file[:11]
        paths.append(os.path.join(split, file))
        labels.append(id_to_label[video_id])

    return paths,labels


# These are lists of paths, so also call load_categories_UCF101(file_path)
# {part}list0x.txt
# Path should end with "/UCF101/ucfTrainTestlist/"
def load_ucf_paths_and_labels(path,split,part):
    partition_file = os.path.join(path, f'{split}list0{part}.txt')
    load_categories_UCF101(path)
    with open(partition_file, 'r') as f:
        all_files = f.read().split('\n')

    paths = list()
    labels = list()

    for file in all_files:
        if file == '':
            continue
        if split == 'train':
            file_name, label = file.split(' ')
            label = int(label) - 1
        else:
            file_name = file
            label = classes[file.split('/')[-1].split('_')[1]]
        paths.append(os.path.join(file_name))
        labels.append(label)

    return paths, labels

def load_files_and_id2label_ssv2(label_path, partition, data_path):
    ### PREPARE PARTITION FILE LIST
    partition_file = os.path.join(label_path, partition + '.json')
    with open(partition_file, 'r') as f:
        partition_data = json.load(f)

    files = [os.path.join(data_path, f"{p['id']}.webm") for p in partition_data]

    ### LOAD LABELS
    labels_file = os.path.join(label_path, 'labels.json')
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    id_to_label = dict()
    if partition != 'test':
        for sample in partition_data:
            # Labels dictionary has first letter as capital letter (not always the case in the partition json)
            id_to_label[sample['id']] = labels[sample['template'].replace("[", "").replace("]", "")]
    else:
        with open(os.path.join(label_path, 'test-answers.csv'), 'r') as f:
            id_to_label = {sample.split(';')[0]: labels[sample.split(';')[1]] for sample in f.read().split('\n')[:-1]}

    return files, id_to_label

def load_k400_id2label(label_path,split):
    with open(os.path.join(label_path, f'{split}.csv'), 'r') as f:
        id_to_label = {sample.split(',')[1]: classes[sample.split(',')[0].replace('"','').replace(' ','_')]
                       for sample in f.read().split('\n')[1:] if sample != ''}
    return id_to_label

def load_categories_K400(file_path):
    global classes
    with open(os.path.join(file_path, 'classes.txt'), 'r') as f:
        text = f.read().split('\n')
        for label, clas in enumerate(text):
            if clas:
                classes[clas] = label


def load_categories_UCF101(file_path):
    global classes
    with open(os.path.join(file_path, 'classInd.txt'), 'r') as f:
        text = f.read().split('\n')
        for line in text:
            if line:
                label, clas = line.split(' ')
                classes[clas] = int(label) - 1



def load_label2name_UCF101(file_path):
    classes = dict()
    with open(os.path.join(file_path, 'classInd.txt'), 'r') as f:
        text = f.read().split('\n')
        for line in text:
            if line:
                label, clas = line.split(' ')
                classes[int(label) - 1] = clas
    return classes


# TODO Inlcude CNN reduction and patch size
def test_possible_downsampling_options(original_size):
    x, y = original_size[0], original_size[1]

    for div in np.arange(2, 5, 0.01):
        d = round(div, 2)
        a, b = x / d, y / d
        if a.is_integer() and b.is_integer():
            print(a, b, d)


def crop_video(vid, final_w, final_h):
    t, h, w, c = vid.shape
    crop_w = (w - final_w) // 2
    crop_h = (h - final_h) // 2
    return vid[:, crop_h:crop_h + final_h, crop_w:crop_w + final_w, :]

def get_video_resolution(video_path):
    command = ["ffprobe",
               "-i", video_path,
               "-select_streams", "v:0",
               "-show_entries", "stream=width,height",
               "-of", "csv=p=0"]
    out = subprocess.run(command, capture_output=True)
    width, height = [int(x) for x in out.stdout.decode('utf-8').split(',')]
    return height, width

def get_video_fps(video_path):
    command = ["ffprobe",
               "-i", video_path,
               "-select_streams", "v:0",
               "-show_entries", "stream=r_frame_rate"]
    out = subprocess.run(command, capture_output=True)
    fps = eval(str(out.stdout).split('\\n')[1].split('=')[1])
    return fps

def set_width_height(height, width, shortest_side, original_h, original_w):
    if width == -1 and height == -1:
        if shortest_side == -1:
            height, width = original_h, original_w
        else:
            h, w = original_h, original_w
            if h < w:
                height = shortest_side
                width = int(w * (height / h))
            else:
                width = shortest_side
                height = int(h * (width / w))
    elif width == -1:
        size_ratio = height / original_h
        width = int(original_w * size_ratio)
    elif height == -1:
        size_ratio = width / original_w
        height = int(original_h * size_ratio)

    return height, width


# Expects x to be a tensor of shape (b,f,h,w,c) and l to be a tensor of shape (b)
def display_data(x, l=None):
    x = x.permute(0, 2, 1, 3, 4) # b,h,f,w,c
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
    plt.imshow(x)
    plt.show()
    # fig, axs = plt.subplots(x.shape[0], figsize=(x.shape[2]//20, x.shape[0]))
    # etiquetas = ""
    # for b in range(x.shape[0]):
    #     etiquetas += "\t " + self.names[l[b].item()]
    #     axs[b].imshow(x[b, :, :, :])
    #     plt.axis('off')
    #     axs[b].tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off',
    #                     left='off', labelleft='off')
    #     plt.draw()
    # print(etiquetas)
    # plt.tight_layout()
    # plt.show()
    # print(len(x))

def display_frame(x):
    #x = x.permute(0,4,2,3,1)
    plt.imshow(x)
    plt.show()

def save_frame(x, path, name):
    # x = x.permute(0,4,2,3,1)
    plt.imshow(x)
    plt.savefig(os.path.join(path, name))
    # plt.show()

if __name__ == '__main__':
    video_path = "/data/ucf101/UCF-101/JumpRope/v_JumpRope_g08_c05.avi"
    print(get_video_fps(video_path))
