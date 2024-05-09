#--------------------------------------------#
#   该部分代码用于看网络结构|计算FLOPS|计算推理速度
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary
import numpy as np

from nets.unet import Unet


def cal_syn(model, device):
    input = torch.randn(1, 3, 224, 224,
                        dtype=torch.float).to(device)  # 创建了随机张量
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True)  # 创建了两个CUDA事件对象starter和ender，用于记录时间
    repetitions = 300  # 循环的迭代次数
    timings = np.zeros((repetitions, 1))  # 每次迭代用的时间，用数组存储

    for _ in range(10):  # 预热GPU，它执行了10次模型前向传播操作，但不会记录时间
        _ = model(input)

    with torch.no_grad():  # 接下来都不计算梯度
        for rep in range(repetitions):  # 进行了repetitions次迭代
            starter.record()  # 记录开始时间
            _ = model(input)  # 正向传播
            ender.record()  # 记录结束时间

            torch.cuda.synchronize()  # 等待GPU同步
            curr_time = starter.elapsed_time(ender)  # 计算耗时
            timings[rep] = curr_time  # 存储在数组中
    mean_syn = np.sum(timings) / repetitions  # 平均时间，也就是推理时间
    std_syn = np.std(timings)  # 时间的标准差
    mean_fps = 1000. / mean_syn  # 每秒处理的帧数
    print(
        ' * Mean@1 {mean_syn:.3f}ms\nStd@5 {std_syn:.3f}ms\nFPS@1 {mean_fps:.2f}'
        .format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
    print(mean_syn)


def cal_FLOPS(model, device):
    input_shape = [512, 512]
    summary(model, (3, input_shape[0], input_shape[1]))

    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(model.to(device), (dummy_input, ), verbose=False)
    # --------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    # --------------------------------------------------------#
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))


if __name__ == "__main__":
    num_classes = 21
    backbone = 'vgg'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(num_classes=num_classes, backbone=backbone).to(device)
    cal_FLOPS(model, device)
    cal_syn(model, device)
