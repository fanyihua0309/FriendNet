import argparse
import os
from tqdm import tqdm
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from data.loader import SingleLoader
from model.dehaze.network import create_model
from model.detect.yolo_test import YOLO
from utils import write_img, chw_to_hwc, pad_img


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='', type=str, required=True)
parser.add_argument('--output_dir', default='', type=str, required=True)
parser.add_argument('--dehaze_model_path', default='checkpoint/FriendNet_best_model.pth', type=str)
parser.add_argument('--detect_model_path', default='checkpoint/yolov7-tiny_clean_best_epoch_weights.pth', type=str)
args = parser.parse_args()


def get_state_dict(model_path):
    state_dict = torch.load(model_path)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict


def test(test_loader, network, yolo):
    print(f'input_dir: {args.input_dir}')
    print(f'output_dir: {args.output_dir}')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    torch.cuda.empty_cache()
    network.eval()
        
    with tqdm(total=len(test_loader), desc='Dehazing') as pbar:
        for idx, batch in enumerate(test_loader):
            input = batch['hazy'].cuda()
            filename = batch['filename'][0]

            with torch.no_grad():
                H, W = input.shape[2:]
                input = pad_img(input, network.patch_size if hasattr(network, 'patch_size') else 16)
                guidance = yolo.get_detection_guidance(input, resize=True).cuda()
                output = network(input, guidance).clamp_(-1, 1)
                output = output[:, :, :H, :W]
                # [-1, 1] to [0, 1]
                output = output * 0.5 + 0.5

            out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
            write_img(os.path.join(args.output_dir, filename), out_img)
            pbar.update(1)


def main():
    network = create_model()
    network = network.cuda()

    if os.path.exists(args.dehaze_model_path):
        print(f'==> Loading pretrained model from {args.dehaze_model_path}')
        network.load_state_dict(get_state_dict(args.dehaze_model_path))
    else:
        print(f'==> No existing pretrained model: {args.dehaze_model_path}')
        exit(0)

    yolo = YOLO(model_path=args.detect_model_path)

    test_dataset = SingleLoader(args.input_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, pin_memory=True)
    test(test_loader, network, yolo)


if __name__ == '__main__':
    main()
