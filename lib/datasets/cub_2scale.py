import os,sys
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_transforms_2scale(cfg):
    interpolation = cfg.DATA.LESSEN_TYPE
    crop_size_large = cfg.DATA.CROP_SIZE
    crop_size_small = int(cfg.DATA.CROP_SIZE/cfg.DATA.LESSEN_RATIO)
    resize_size_small = int(cfg.DATA.RESIZE_SIZE / cfg.DATA.LESSEN_RATIO)

    train_transform_large = transforms.Compose([
        transforms.RandomResizedCrop(crop_size_large),
        transforms.RandomHorizontalFlip(),
    ])

    train_transform_small = transforms.Compose([
        transforms.Resize((crop_size_small, crop_size_small), interpolation=interpolation)
    ])

    train_transform_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(resize_size_small),
        transforms.CenterCrop(crop_size_small),
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD)
    ])
    return train_transform_large, train_transform_small, train_transform_normalize, test_transform


class CUBDataset(Dataset):
    def __init__(self, cfg, is_train):

        self.root = os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATA.DATADIR)
        self.cfg = cfg
        self.is_train = is_train
        self.resize_size = cfg.DATA.RESIZE_SIZE
        self.crop_size = cfg.DATA.CROP_SIZE

        self.image_list = self.remove_1st_column(open(
            os.path.join(self.root, 'images.txt'), 'r').readlines())
        self.label_list = self.remove_1st_column(open(
            os.path.join(self.root, 'image_class_labels.txt'), 'r').readlines())
        self.split_list = self.remove_1st_column(open(
            os.path.join(self.root, 'train_test_split.txt'), 'r').readlines())
        self.bbox_list = self.remove_1st_column(open(
            os.path.join(self.root, 'bounding_boxes.txt'), 'r').readlines())

        self.train_transform_large, self.train_transform_small, \
        self.train_transform_normalize, self.test_transform = get_transforms_2scale(cfg)

        if is_train:
            self.index_list = self.get_index(self.split_list, '1')
        else:
            self.index_list = self.get_index(self.split_list, '0')

    def get_index(self, list, value):
        index = []
        for i in range(len(list)):
            if list[i] == value:
                index.append(i)
        return index

    def remove_1st_column(self, input_list):
        output_list = []
        for i in range(len(input_list)):
            if len(input_list[i][:-1].split(' '))==2:
                output_list.append(input_list[i][:-1].split(' ')[1])
            else:
                output_list.append(input_list[i][:-1].split(' ')[1:])
        return output_list

    def __getitem__(self, idx):
        name = self.image_list[self.index_list[idx]]
        image_path = os.path.join(self.root, 'images', name)
        image = Image.open(image_path).convert('RGB')
        target = int(self.label_list[self.index_list[idx]])-1

        if self.is_train:
            image_large = self.train_transform_large(image)
            toTensor = transforms.ToTensor()
            image_large_tensor = toTensor(image_large)
            if self.cfg.DATA.LESSEN_RATIO == 1:
                image_large = self.train_transform_normalize(image_large)
                image_small = image_large
            else:
                image_small = self.train_transform_small(image_large)
                image_large = self.train_transform_normalize(image_large)
                image_small = self.train_transform_normalize(image_small)
            return image_large, image_small, target, image_large_tensor
        else:
            image = self.test_transform(image)
            return image, target, name[:-4]

        # x = transforms.ToTensor()
        # a = x(image_large)

    def __len__(self):
        return len(self.index_list)


if __name__ == "__main__":
    import os, sys

    sys.path.insert(0, '../../lib')

    import argparse
    from config.default import update_config
    from config.default import config as cfg

    parser = argparse.ArgumentParser(description='knowledge distillation')
    parser.add_argument('--config_file', type=str, default='../../configs/cub/cub_resnet50_resnet_kd.yaml',
                        required=False, help='Optional config file for params')
    parser.add_argument('opts', help='see config.py for all options',
                        default='BASIC.GPU_ID [7]', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(args)

    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..')

    import torch

    train_loader = torch.utils.data.DataLoader(
        CUBDataset(cfg=cfg, is_train=True),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CUBDataset(cfg=cfg, is_train=False),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    for image_l, image_s, label, name in train_loader:
        print(label)
        print(image_l.shape)
        print(image_s.shape)
        print(name)
        break

    for image, label, name in val_loader:
        print(label)
        print(image.shape)
        print(name)
        break









