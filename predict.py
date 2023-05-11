import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    #  net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    # resize img
    # img转换成pytorch的输入格式（维度转换）
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        # logits = self.outc(x)
        # return logits
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        # full_img.size[1], full_img.size[0]), which is the height and width of the input image. T
        # 上采样，bilinear插值  
        if net.n_classes > 1:
         # args.classes default=2
            mask = output.argmax(dim=1)  
        #  taking the index of the maximum value along the second dimension of the output tensor.
        #  img 数组沿着第一个维度（即通道维度）进行红、绿、蓝三个通道上每个像素值的 argmax 操作。
        #  它将返回一个形状为 (256, 256) 的数组，表示每个像素最可能的颜色通道
        else:
            mask = torch.sigmoid(output) > out_threshold 
        # 二值化
        # hich returns a tensor of the same shape as output with boolean values 
        # indicating whether each element is greater than the threshold
    return mask[0].long().squeeze().numpy()
    # 图像的第一行在颜色通道上取得最大值时对应的通道编号。


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoint_epoch5.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    # default='MODEL.pth'
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    # --input 和 -i：这两个字符串分别表示长选项和短选项（也叫简写选项）的名称，用于在命令行中指定参数。
    # metavar='INPUT'：这个字符串表示参数值的占位符，用于在帮助信息中描述参数的用途。在这个例子中，使用 INPUT 作为占位符，表示这个参数需要指定一个或多个输入文件名。
    # nargs='+'：这个参数表示输入的文件名可以有一个或多个，即支持多个文件名同时输入。
    # help='Filenames of input images'：这个字符串表示在使用 -h 或 --help 显示帮助信息时需要显示的参数描述。
    # required=True：这个参数表示这个参数是必须的，没有提供该参数时将会抛出错误。

    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    # default=2
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    # mask_values 的第一个元素是一个列表或数组，则说明每个标签有多个取值。
    # 在这种情况下，输出数组 out 的第三个维度等于 len(mask_values[0])，
    # 即每个像素的值需要使用多个通道来表示不同标签的取值。
    #mask.shape[-2] 和 mask.shape[-1] 分别表示输入 mask 的倒数第二个和倒数第一个维度的长度，即高度和宽度。
    # 因此，out 的形状为 (height, width, num_classes)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    # input参数
    out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    # change channel here default = 3
    # args.classes default=2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)  
    # 加载参数
    # default='MODEL.pth'
    mask_values = state_dict.pop('mask_values', [0, 1]) 
     # 移动字典里的 mask_values值到mask_values，没有的话就默认值[0,1]
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)  #
