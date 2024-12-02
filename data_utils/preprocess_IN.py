from PIL import Image
import os

def resize_image(img, shortest_side):
    width, height = img.size
    if width < height:
        ratio = shortest_side / width
        new_width = shortest_side
        new_height = int(height * ratio)
    else:
        ratio = shortest_side / height
        new_height = shortest_side
        new_width = int(width * ratio)
    img = img.resize((new_width, new_height), Image.BICUBIC)
    return img

# Given a path to a folder, load all images one by one, resize it according to the shortest_side and save it again
def prepare_partition(save_path, data_path, shortest_side):
    for clas in os.listdir(data_path):
        class_path = os.path.join(data_path, clas)
        out_class_path = os.path.join(save_path, clas)
        if not os.path.exists(out_class_path):
            os.mkdir(out_class_path)
        for image in os.listdir(class_path):
            image_path = os.path.join(data_path, clas, image)
            out_image_path = os.path.join(save_path, clas, image)
            if not os.path.isfile(image_path) or os.path.exists(out_image_path):
                continue
            with Image.open(image_path) as img:
                img = resize_image(img, shortest_side)
                if img.mode != 'RGB':
                    print(image_path)
                    img = img.convert('RGB')
                img.save(out_image_path)


if __name__ == '__main__':
    #### PREPARE CSV FILE
    data_path = "/data-net/datasets/ImageNet-1k/train"
    out_dir_base = "/data-net/hupba/jselva/imagenet/train"
    prepare_partition(out_dir_base, data_path, 256)
    data_path = "/data-net/datasets/ImageNet-1k/val"
    out_dir_base = "/data-net/hupba/jselva/imagenet/val"
    prepare_partition(out_dir_base, data_path, 256)
