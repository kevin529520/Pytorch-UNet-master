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
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    # glob迭代器，访问mask_dir路径下的文件（.*表示不限文件格式）list:[WindowsPath('data/masks/00087a6bd4dc_01_mask.gif')]
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
    # 只有这一段是自己加的，相当于手动给图片做二值化（加了这段之后图片加载变慢，之后还得修改）
    # 数组二值化
    # print(mask)
    if mask.ndim == 2:
        return np.unique(mask_binary)
        # mask数组中所有不一样的数
    # 灰白图像 返回不同像素值 一个一维数组
    elif mask.ndim == 3:
        mask_binary = mask_binary.reshape(-1, mask_binary.shape[-1])
        # 保留列数不变，变换成二维
        return np.unique(mask_binary, axis=0)
        # 第一维（即行）进行计算，也就是说这个函数将会保留并返回在 mask 中出现的所有不同的行，而去除重复的行。
    # 彩色图像 返回不同行 一个二维数组
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

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if
                    isfile(join(images_dir, file)) and not file.startswith('.')]  # id= '0495dcf27283_16',
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))  # partial将unique_mask_values的参数固定   p.imap多个进程并行地将函数unique_mask_values应用于可迭代对象self.ids中的每个元素。
        # unique_mask_values 一维数组
        # unique 二维数组的 列表
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        # 灰度图片
        # unique 二维数组的 列表([[1,3],[5,4],[2,3],[1,6]])
        # np.concatenate(unique) 拼接成矩阵[1 3 5 4 2 3 1 6]
        # np.unique 去重 保留[1, 2, 3, 4, 5, 6]
        # 转换为 Python 中的列表类型 sorted 函数根据这些不同的行的值进行排序[1, 2, 3, 4, 5, 6]
        # unique 中的二维数组进行拼接时，各个数组的列数必须相同，否则将会抛出 ValueError 异常。

        # 彩色图片
        # unique 三维数组的 列表[[[1,2],[5,3],[2,3],[1,6]],
        #                     [[1,3],[5,4],[2,3],[1,6]],]
        # np.concatenate(unique) 拼接成一个二维矩阵
        # np.unique axis=0 去除同样的行，保留 [[1, 2], [1, 3], [1, 6], [2, 3], [5, 3], [5, 4]]
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
                # i 表示当前子列表的索引，v 表示当前子列表的值。
                # 灰度 mask_value 一维数组，数字代表每个像素不一样的颜色
                # 彩色 mask_value 二维数组，每行代表原来图片不同行
                # array([[0.11869271, 0.79792323, 0.68079955, 0.19730384, 0.88891563],
                #        [0.29497223, 0.96196393, 0.02844176, 0.25685066, 0.53844648],
                #        [0.35520807, 0.5269884 , 0.57979242, 0.65758224, 0.58410875],
                #        [0.42392343, 0.58959185, 0.48060784, 0.92627966, 0.4081902 ],
                #        [0.50808203, 0.72856992, 0.16289775, 0.12744291, 0.73791519],
                #        [0.62380536, 0.16354594, 0.82131196, 0.75408024, 0.80766366],
                #        [0.69936868, 0.28317407, 0.0937293 , 0.91471367, 0.19398882],
                #        [0.71815255, 0.47311478, 0.03951405, 0.19968953, 0.84733757],
                #        [0.74185751, 0.66206757, 0.80533779, 0.84880646, 0.48741714],
                #        [0.84449748, 0.43453824, 0.87280792, 0.42546177, 0.94912969],
                #        [0.87660732, 0.98860549, 0.75781347, 0.13276018, 0.18412097],
                #        [0.99543955, 0.7660869 , 0.58871765, 0.23138931, 0.27072119]])
                if img.ndim == 2:
                    mask[img == v] = i
                # v与i的映射
                # 对于二维的图片，通过img == v得到一个与img具有相同形状的布尔数组，其中元素为True表示该位置上的像素值等于v（或者数组v中一个元素就行）。
                # 然后将该布尔数组作为索引，对mask数组中对应位置赋值为i，即将具有相同颜色的像素标记为相同的值。
                else:
                    mask[(img == v).all(-1)] = i
                # all(-1) 方法对比较结果按最后一维求和，得到一个布尔类型的数组，形状为 (3, 4)，其中第 (i, j) 个元素表示 img[i, j, :] 是否与 v 相等。
                # 最后，将 mask 数组中对应位置为 True 的元素赋值为 i。
                # mask 数组是一个与 img 具有相同形状的数组，其中的元素代表了该位置上的像素属于哪个颜色类别，即进行了像素的分割操作。

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
                # 加一个维度 batch_size
            else:
                img = img.transpose((2, 0, 1))
            # 这行代码将图像数组的维度从(0,1,2)转换为(2,0,1)，即将通道维度放在第一维，
            # 这是因为在PyTorch中，卷积层的输入是(batch_size, channels, height, width)。
            # 因此，这行代码将图像数组转换为PyTorch所需的格式。

            if (img > 1).any():
                img = img / 255.0
            # (img > 1).any() 是一个检查条件，首先比较 img 中的每个元素是否大于 1，产生一个布尔数组，
            # 然后用 .any() 方法判断这个数组中是否有至少一个元素为 True。
            # 如果有，则说明 img 中存在像素值超过 1 的情况，需要进行归一化。

            # 在深度学习中，对于输入数据的范围，归一化通常是一种常见的预处理方式。
            # 归一化能够使所有特征的权重具有相同的量纲，避免模型因为某些特征取值较大而产生过度拟合的情况。
            # 同时，归一化也能够加速模型的训练收敛速度，有助于提高模型的精度和鲁棒性。
            # 将像素值缩放到 0～1 的范围内，有助于加速梯度下降算法的收敛过程，避免数据范围过大或过小造成的梯度消失或爆炸的问题。

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]  # imgs和masks是同一个name
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))  # list，但只有一个值
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])  # convert to array
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        # 预处理 scale  以及像素的归一化
        # 对于灰度图像 放缩 加batch_size维度                                      像素归一化
        # 对于彩色图像 放缩 将数组的维度从(0,1,2)转换为(2,0,1)，即将通道维度放在第一维，  像素归一化
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        # 对于灰度图像 放缩 像素值按从小到大映射到1，2，3，4，5...
        # 对于彩色图像 放缩 像素 组 （？）按从小到大映射到1，2，3，4，5...
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
        # 返回一个字典
        # torch.as_tensor() 无需复制数据，可以加速数据传输和减少内存使用。
        # 使用 float() 方法将其转换为浮点数型张量，并使用 contiguous() 方法使得张量在内存中连续排列，以提高计算效率


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale,
                         mask_suffix='_mask')  # super().__init__()调用了BasicDataset类的构造函数，这样Car..类就可以继承Bas..类的所有属性和方法。
# if __name__ == '__main__':
#     dir_img = Path('../data/imgs/')
#     dir_mask = Path('../data/binary_masks/')
#     img_scale=1
#     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
