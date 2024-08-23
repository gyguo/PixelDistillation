import os
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




class AircraftDataset(Dataset):
    def __init__(self, cfg, is_train):

        self.root = os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATA.DATADIR)
        self.cfg = cfg
        self.is_train = is_train
        self.resize_size = cfg.DATA.RESIZE_SIZE
        self.crop_size = cfg.DATA.CROP_SIZE

        self.image_folder = os.path.join(self.root, "data", "images")
        self.train_transform_large, self.train_transform_small, \
        self.train_transform_normalize, self.test_transform = get_transforms_2scale(cfg)
        # "train", "val", "trainval", "test"
        if is_train:
            self._split = "trainval"
        else:
            self._split = "test"

        self._annotation_level = "variant"  # "variant", "family", "manufacturer"

        annotation_file = os.path.join(
            self.root,
            "data",
            {
                "variant": "variants.txt",
                "family": "families.txt",
                "manufacturer": "manufacturers.txt",
            }[self._annotation_level],
        )
        with open(annotation_file, "r") as f:
            self.classes = [line.strip() for line in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        labels_file = os.path.join(self.root, "data", f"images_{self._annotation_level}_{self._split}.txt")

        self._image_files = []
        self._labels = []

        with open(labels_file, "r") as f:
            for line in f:
                image_name, label_name = line.strip().split(" ", 1)
                self._image_files.append(os.path.join(self.image_folder, f"{image_name}.jpg"))
                self._labels.append(self.class_to_idx[label_name])

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, idx):
        image_file, target = self._image_files[idx], self._labels[idx]
        image = Image.open(image_file).convert("RGB")

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
            return image, target, image_file


if __name__ == "__main__":
    import os, sys

    sys.path.insert(0, '../../lib')

    import argparse
    from config.default import update_config
    from config.default import config as cfg

    parser = argparse.ArgumentParser(description='knowledge distillation')
    parser.add_argument('--config_file', type=str, default='../../configs/aircraft/aircraft_resnet50_32x4d_resnet_kd.yaml',
                        required=False, help='Optional config file for params')
    parser.add_argument('opts', help='see config.py for all options',
                        default='BASIC.GPU_ID [7]', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(args)

    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..')

    import torch

    train_loader = torch.utils.data.DataLoader(
        AircraftDataset(cfg=cfg, is_train=True),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        AircraftDataset(cfg=cfg, is_train=False),
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