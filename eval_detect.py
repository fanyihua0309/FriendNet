import argparse
import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import shutil
from model.detect.yolo_test import YOLO
from utils.utils_detect import get_classes
from utils.utils_map import get_map


parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='', type=str, required=True)
parser.add_argument('--label_dir', default='', type=str, required=True)
parser.add_argument('--detect_model_path', default='checkpoint/yolov7-tiny_clean_best_epoch_weights.pth', type=str)
parser.add_argument('--map_mode', default=0, type=int,
                    help='0: entire mAP computing process (including 1, 2, 3); \
                    1: get predicted results; \
                    2: get ground truth results; \
                    3: compute mAP@0.5')
parser.add_argument('--classes_path', default='data/voc_classes.txt', type=str)
parser.add_argument('--map_out_path', default='map_out', type=str)
args = parser.parse_args()


def main():
    print(f'image_dir: {args.image_dir}')
    print(f'laber_dir: {args.label_dir}')

    ext = os.listdir(args.image_dir)[0].split('.')[-1]
    image_ids = [os.path.basename(img_name).split(f'.{ext}')[0] for img_name in os.listdir(args.image_dir)]
    class_names, _ = get_classes(args.classes_path)

    map_out_path = args.map_out_path
    # make sure the mAP out path is empty when starting a new evaluation process
    if os.path.exists(map_out_path):
        shutil.rmtree(map_out_path)
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    map_mode = args.map_mode

    if map_mode == 0 or map_mode == 1:
        yolo = YOLO(model_path=args.detect_model_path, confidence=0.001, nms_iou=0.5)

        with tqdm(total=len(image_ids), desc='Get predicted results') as pbar:
            for image_id in image_ids:
                image_path = os.path.join(args.image_dir, image_id + f'.{ext}')
                image = Image.open(image_path)
                yolo.get_map_txt(image_id, image, class_names, map_out_path)
                pbar.update(1)

    if map_mode == 0 or map_mode == 2:
        with tqdm(total=len(image_ids), desc='Get ground truth results') as pbar:
            for image_id in image_ids:
                with open(os.path.join(map_out_path, 'ground-truth/' + image_id + '.txt'), 'w') as new_f:
                    root = ET.parse(os.path.join(args.label_dir, image_id + '.xml')).getroot()
                    for obj in root.findall('object'):
                        difficult_flag = False
                        if obj.find('difficult') != None:
                            difficult = obj.find('difficult').text
                            if int(difficult) == 1:
                                difficult_flag = True
                        obj_name = obj.find('name').text
                        if obj_name not in class_names:
                            continue
                        bndbox = obj.find('bndbox')
                        left = bndbox.find('xmin').text
                        top = bndbox.find('ymin').text
                        right = bndbox.find('xmax').text
                        bottom = bndbox.find('ymax').text

                        if difficult_flag:
                            new_f.write('%s %s %s %s %s difficult\n' % (obj_name, left, top, right, bottom))
                        else:
                            new_f.write('%s %s %s %s %s\n' % (obj_name, left, top, right, bottom))
                pbar.update(1)

    if map_mode == 0 or map_mode == 3:
        print('Get mAP.')
        print('-' * 100)
        get_map(MINOVERLAP=0.5, draw_plot=True, score_threhold=0.5, path=map_out_path)
        print('-' * 100)
        print('Get mAP done.')


if __name__ == '__main__':
    main()
