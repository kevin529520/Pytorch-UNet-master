import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):  # 加载图片，在unique_mask_values中的 mask = np.asarray(load_image(mask_file))调用
    ext = splitext(filename)[1]  # 后缀
    if ext == '.npy':
        return Image.fromarray(np.load(filename))  # 将数组转换成图像
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):  # suffix   imgs到mask的value转变，加suffix
    # logging.info('进入unique函数')
    # print('1')
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]  # glob迭代器，访问mask_dir路径下的文件（.*表示不限文件格式）list:[WindowsPath('data/masks/00087a6bd4dc_01_mask.gif')]
    # logging.info('已执行list(mask_dir.glob....)')
    # print('2')
    mask = np.asarray(load_image(mask_file))
    mask_binary = np.zeros_like(mask)

    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] > 127:
                mask_binary[i][j] = 255
            else:
                mask_binary[i][j] = 0
    # 数组二值化
    # print(mask)
    if mask.ndim == 2:
        return np.unique(mask_binary)
    elif mask.ndim == 3:
        mask_binary = mask_binary.reshape(-1, mask_binary.shape[-1])
        return np.unique(mask_binary, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
    # with Pool() as p:
    #     unique = list(tqdm(
    #         p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
    #         total=len(self.ids)
    #     ))  # partial将unique_mask_values的参数固定   p.imap多个进程并行地将函数unique_mask_values应用于可迭代对象self.ids中的每个元素。
    #
    # self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)  # './data/imgs/'
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix  # '_mask'

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]  # id= '0495dcf27283_16',
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))  # partial将unique_mask_values的参数固定   p.imap多个进程并行地将函数unique_mask_values应用于可迭代对象self.ids中的每个元素。

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)
        # resize img

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))
            #这行代码将图像数组的维度从(0,1,2)转换为(2,0,1)，即将通道维度放在第一维，
            # 这是因为在PyTorch中，卷积层的输入是(batchsize, channels, height, width)。
            # 因此，这行代码将图像数组转换为PyTorch所需的格式。

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]#imgs和masks是同一个name
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*')) #list，但只有一个值
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0]) #convert to array
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)#预处理 scale
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')#super().__init__()调用了BasicDataset类的构造函数，这样Car..类就可以继承Bas..类的所有属性和方法。
