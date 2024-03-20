import argparse
import os
from tqdm import tqdm
from PIL import Image
import torch
from torchvision.transforms import transforms
from metric.psnr import compute_psnr
from metric.ssim import compute_ssim
from utils import AverageMeter


parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='', type=str, required=True)
parser.add_argument('--gt_dir', default='', type=str, required=True)
args = parser.parse_args()


def load_image(img_path):
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert('RGB')
    img = trans(img).unsqueeze(0)
    img = img.cuda()
    return img


def compute_metrics():
    print(f'image_dir: {args.image_dir}')
    print(f'gt_dir: {args.gt_dir}')

    psnr = AverageMeter()
    ssim = AverageMeter()
    with tqdm(total=len(os.listdir(args.image_dir)), desc='Eval image quality') as pbar:
        with torch.no_grad():
            for img_name in os.listdir(args.image_dir):
                img_path = os.path.join(args.image_dir, img_name)
                img = load_image(img_path)
                clear_img_path = os.path.join(args.gt_dir, img_name)
                clear_img = load_image(clear_img_path)

                cur_psnr = compute_psnr(img, clear_img).item()
                cur_ssim = compute_ssim(img, clear_img).item()
                psnr.update(cur_psnr)
                ssim.update(cur_ssim)
                pbar.set_postfix(**{
                    'PSNR': f'{psnr.avg:.4f}',
                    'SSIM': f'{ssim.avg:.4f}',
                })
                pbar.update(1)
            print(f'PSNR: {psnr.avg}   SSIM: {ssim.avg}')


if __name__ == '__main__':
    compute_metrics()
