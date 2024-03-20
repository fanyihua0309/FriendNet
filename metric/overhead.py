import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from thop import profile
import time
from model.dehaze.network import create_model


if __name__ == '__main__':
    model = create_model().cuda()
    input = torch.randn((1, 3, 256, 256)).cuda()
    guidance = torch.randn((1, 3, 256, 256)).cuda()

    print('==> Building model..')
    flops, params = profile(model, (input, guidance))
    print(f'Parameters: {params / 1e6:.2f} M')
    print(f'FLOPs: {flops / 1e9:.2f} G')

    # Runtime
    model.eval()

    # 预热模型（可选，确保模型已被加载到GPU等设备）
    with torch.no_grad():
        model(input, guidance)

    # 可以根据实际情况调整迭代次数
    num_iterations = 100
    total_time = 0.0

    for _ in range(num_iterations):
        start_time = time.time()

        with torch.no_grad():
            output = model(input, guidance)

        end_time = time.time()
        total_time += (end_time - start_time)

    # 计算平均推理时间
    average_inference_time = total_time / num_iterations

    print(f'Runtime: {average_inference_time * 1e3:.2f} ms')