import argparse
import os
from PIL import Image
from tqdm import tqdm
from model.detect.yolo_test import YOLO


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='', type=str, required=True)
parser.add_argument('--output_dir', default='', type=str, required=True)
parser.add_argument('--detect_model_path', default='checkpoint/yolov7-tiny_clean_best_epoch_weights.pth', type=str)
args = parser.parse_args()


def save_predicted_results():
    print(f'input_dir: {args.input_dir}')
    print(f'output_dir: {args.output_dir}')
    os.makedirs(args.output_dir, exist_ok=True)
    img_names = os.listdir(args.input_dir)

    yolo = YOLO(model_path=args.detect_model_path, confidence=0.5, nms_iou=0.3)

    with tqdm(total=len(img_names), desc='Save predicted results') as pbar:
        for img_name in img_names:
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(args.input_dir, img_name)
                image = Image.open(image_path)
                r_image = yolo.detect_image(image)
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                r_image.save(os.path.join(args.output_dir, img_name), quality=95, subsampling=0)
                pbar.update(1)


if __name__ == '__main__':
    save_predicted_results()
