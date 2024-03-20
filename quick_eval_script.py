import os


# mode type
# 0: entire process (including 1, 2, 3)
# 1: test dehaze
# 2: eval dehaze
# 3: eval detect
mode = 0

# 'VOC-FOG' 'Foggy_Driving'
dataset_name = 'VOC-FOG'
method_name = 'Ours'

hazy_dir = f'data/{dataset_name}/test/FogImages'
dehazed_dir = f'data/{dataset_name}/test/SOTAs/{method_name}'
gt_dir = f'data/{dataset_name}/test/JPEGImages'
label_dir = f'data/{dataset_name}/test/Annotations'
map_out_path = r'map_out'

dehaze_model_path = 'checkpoint/FriendNet_best_model.pth'
detect_model_path = 'checkpoint/yolov7-tiny_clean_best_epoch_weights.pth'


if mode == 0 or mode == 1:
    os.system(f'python test_dehaze.py --input_dir={hazy_dir} --output_dir={dehazed_dir} '
              f'--dehaze_model_path={dehaze_model_path} --detect_model_path={detect_model_path}')
if (mode == 0 or mode == 2) and dataset_name == 'VOC-FOG':
    os.system(f'python eval_dehaze.py --image_dir={dehazed_dir} --gt_dir={gt_dir}')
if mode == 0 or mode == 3:
    os.system(f'python eval_detect.py --image_dir={dehazed_dir} --label_dir={label_dir} '
              f'--map_out_path={map_out_path} '
              f'--detect_model_path={detect_model_path}')
