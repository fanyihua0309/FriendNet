import os
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm
from utils.utils_detect import get_classes


classes_path = 'data/voc_classes.txt'
classes, _ = get_classes(classes_path)
data_dir_name = {'fog': 'FogImages', 'clear': 'JPEGImages', 'label': 'Annotations'}


# get class and location coordinates from xml file
def convert_annotation(in_file_path, out_file, class_nums):
    in_file = open(in_file_path)
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        class_nums[cls_id] += 1
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        out_file.write(' ' + ','.join([str(a) for a in b]) + ',' + str(cls_id))


# generate txt file
def generate_txt(annonation_dir, txt_dir, phase='test'):
    out_file_path = os.path.join(txt_dir, f'{phase}.txt')
    out_file = open(out_file_path, 'w')
    class_nums = [0 for cls in classes]
    total_num = len(os.listdir(annonation_dir))
    with tqdm(total=total_num, desc='Convert test') as pbar:
        for file_name in os.listdir(annonation_dir):
            in_file_path = os.path.join(annonation_dir, file_name)
            img_path = in_file_path.replace(data_dir_name['label'], data_dir_name['fog']).replace('xml', 'jpg')
            img_path = img_path.replace(os.path.sep, '/')
            out_file.write(img_path)
            out_file.write(' ')
            img_path = in_file_path.replace(data_dir_name['label'], data_dir_name['clear']).replace('xml', 'jpg')
            img_path = img_path.replace(os.path.sep, '/')
            out_file.write(img_path)
            convert_annotation(in_file_path, out_file, class_nums)
            out_file.write('\n')
            pbar.update(1)
    out_file.close()

    print(f'=> Test Data: {total_num}')
    for i in range(len(classes)):
        print(f'{classes[i]}: {class_nums[i]}', end='   ')


# if you want to split the training set into two parts (train and val), you can use this function
# generate train.txt and val.txt for training (random split train set)
def generate_train_val_txt(annonation_dir, txt_dir, train_ratio=0.9):
    train_file_path = os.path.join(txt_dir, 'train.txt')
    val_file_path = os.path.join(txt_dir, 'val.txt')
    train_file = open(train_file_path, 'w')
    val_file = open(val_file_path, 'w')
    train_class_nums = [0 for cls in classes]
    val_class_nums = [0 for cls in classes]

    total_num = len(os.listdir(annonation_dir))
    train_num = int(total_num * train_ratio)
    val_num = total_num - train_num
    val_set = random.sample(os.listdir(annonation_dir), val_num)

    with tqdm(total=total_num, desc='Convert train and val') as pbar:
        for file_name in os.listdir(annonation_dir):
            in_file_path = os.path.join(annonation_dir, file_name)

            img_path = in_file_path.replace(data_dir_name['label'], data_dir_name['fog']).replace('xml', 'jpg')
            img_path = img_path.replace(os.path.sep, '/')
            img_path_2 = in_file_path.replace(data_dir_name['label'], data_dir_name['clear']).replace('xml', 'jpg')
            img_path2 = img_path2.replace(os.path.sep, '/')
            if file_name in val_set:
                val_file.write(img_path)
                val_file.write(' ')
                val_file.write(img_path_2)
                convert_annotation(in_file_path, val_file, val_class_nums)
                val_file.write('\n')
            else:
                train_file.write(img_path)
                train_file.write(' ')
                train_file.write(img_path_2)
                convert_annotation(in_file_path, train_file, train_class_nums)
                train_file.write('\n')

            pbar.update(1)
    train_file.close()
    val_file.close()

    print(f'=> Train Data: {train_num}')
    for i in range(len(classes)):
        print(f'{classes[i]}: {train_class_nums[i]}', end='   ')

    print(f'\n=> Val Data: {val_num}')
    for i in range(len(classes)):
        print(f'{classes[i]}: {val_class_nums[i]}', end='   ')


if __name__ == '__main__':
    txt_dir = 'data'
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)

    annonation_dir = 'data/VOC-FOG/train/Annotations'
    generate_txt(annonation_dir, txt_dir, phase='train')

    annonation_dir = 'data/VOC-FOG/test/Annotations'
    generate_txt(annonation_dir, txt_dir, phase='test')
