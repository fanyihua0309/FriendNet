import os
from PIL import Image
import shutil


root = r'data/Foggy_Driving'
img_dir_name = 'leftImg8bit'
txt_dir_name = 'bboxGt'
dst_img_dir_name = 'FOGImages'
dst_txt_dir_name = 'Annotations'
dst_img_dir = os.path.join(root, dst_img_dir_name)
if not os.path.exists(dst_img_dir):
    os.mkdir(dst_img_dir)

dst_txt_dir = os.path.join(root, dst_txt_dir_name)
if not os.path.exists(dst_txt_dir):
    os.mkdir(dst_txt_dir)
classes = {
    '0': 'car',
    '1': 'person',
    '2': 'bicycle',
    '3': 'bus',
    '4': 'truck',
    '5': 'train',
    # to ensure consistency, we use 'motorbike' instead of 'motorcycle'
    '6': 'motorbike',
    '7': 'rider',
}
img_dir = os.path.join(root, img_dir_name)
for dir_name in os.listdir(img_dir):
    dir_path = os.path.join(img_dir, dir_name)
    for sub_dir_name in os.listdir(dir_path):
        sub_dir_path = os.path.join(dir_path, sub_dir_name)
        for img_name in os.listdir(sub_dir_path):
            print(img_name)
            save_img_name = img_name.split('_leftImg8bit.png')[0] + '.png'
            img_path = os.path.join(sub_dir_path, img_name)
            txt_path = os.path.join(sub_dir_path.replace(img_dir_name, txt_dir_name), save_img_name.replace('png', 'txt'))
            shutil.copyfile(img_path, os.path.join(dst_img_dir, save_img_name))

            img = Image.open(img_path)
            width, height = img.size
            gt = open(txt_path).read().splitlines()
            xml_file = open(os.path.join(dst_txt_dir, save_img_name.replace('png', 'xml')), 'w')
            xml_file.write('<annotation>\n')
            xml_file.write('    <folder>VOC2007</folder>\n')
            xml_file.write('    <filename>' + str(img_name) + '.png' + '</filename>\n')
            xml_file.write('    <size>\n')
            xml_file.write('        <width>' + str(width) + '</width>\n')
            xml_file.write('        <height>' + str(height) + '</height>\n')
            xml_file.write('        <depth>3</depth>\n')
            xml_file.write('    </size>\n')

            # write the region of image on xml file
            for img_each_label in gt:
                spt = img_each_label.split(' ')
                xml_file.write('    <object>\n')
                xml_file.write('        <name>' + str(classes[spt[0]]) + '</name>\n')
                xml_file.write('        <pose>Unspecified</pose>\n')
                xml_file.write('        <truncated>0</truncated>\n')
                xml_file.write('        <difficult>0</difficult>\n')
                xml_file.write('        <bndbox>\n')
                xml_file.write('            <xmin>' + str(spt[1]) + '</xmin>\n')
                xml_file.write('            <ymin>' + str(spt[2]) + '</ymin>\n')
                xml_file.write('            <xmax>' + str(spt[3]) + '</xmax>\n')
                xml_file.write('            <ymax>' + str(spt[4]) + '</ymax>\n')
                xml_file.write('        </bndbox>\n')
                xml_file.write('    </object>\n')

            xml_file.write('</annotation>')