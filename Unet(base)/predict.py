import os
import time

import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from train import parse_args,create_model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():

    args = parse_args()
    #测试的目录
    # test_path = 'data/val/images/'
    test_path = 'data/test/images/'
    save_path = 'predictions/'

    #加载权重
    weights_path = "./save_weights/best_model.pth"

    #均值和方差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model(args.num_classes).to(device)

    # 加载模型
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(os.path.join(save_path))
    model.eval()  # 进入验证模式
    with torch.no_grad():
        #对test目录下的图片进行测试
        for i in os.listdir(test_path):
            # load image
            img_path = os.path.join(test_path,i)
            original_img = Image.open(img_path).convert('RGB')

            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])

            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))
            pred = output.argmax(1).squeeze(0).to("cpu").numpy().astype(np.uint8)

            # 将前景对应的像素值改成某个值
            pred[pred == 1] = 255

            mask = Image.fromarray(pred)
            mask.save(os.path.join(save_path,i))


if __name__ == '__main__':
    main()
