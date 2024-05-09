import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()

        self.flag = "train" if train else "val"
        data_root = os.path.join(root, self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms


        self.img_list = [os.path.join(data_root, "images", i) for i in
                         os.listdir(os.path.join(data_root, "images"))]


    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        label_fp = self.img_list[idx].replace('images','labels')
        mask = Image.open(label_fp).convert('L')
        mask = np.array(mask)
        mask = mask / 255
        mask[mask>=0.5] = 1
        mask[mask<0.5] = 0


        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask

    def __len__(self):
        return len(self.img_list)
