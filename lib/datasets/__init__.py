import torch


def creat_data_loader(cfg):
    print('==> Preparing data...')
    if cfg.DATA.DATASET == 'cub':
        from .cub import CUBDataset
        train_loader = torch.utils.data.DataLoader(
            CUBDataset(cfg=cfg, is_train=True), batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=True, num_workers=cfg.BASIC.NUM_WORKERS, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            CUBDataset(cfg=cfg, is_train=False), batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False, num_workers=cfg.BASIC.NUM_WORKERS, pin_memory=True)
    elif cfg.DATA.DATASET == 'aircraft':
        from .aircraft import AircraftDataset
        train_loader = torch.utils.data.DataLoader(
            AircraftDataset(cfg=cfg, is_train=True), batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=True, num_workers=cfg.BASIC.NUM_WORKERS, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            AircraftDataset(cfg=cfg, is_train=False), batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False, num_workers=cfg.BASIC.NUM_WORKERS, pin_memory=True)
    else:
        raise ValueError('Please set correct dataset.')

    print('done!')
    return train_loader, val_loader


def creat_data_loader_2scale(cfg):
    print('==> Preparing data...')
    if cfg.DATA.DATASET == 'cub':
        from .cub_2scale import CUBDataset
        train_loader = torch.utils.data.DataLoader(
            CUBDataset(cfg=cfg, is_train=True), batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=True, num_workers=cfg.BASIC.NUM_WORKERS, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            CUBDataset(cfg=cfg, is_train=False), batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False, num_workers=cfg.BASIC.NUM_WORKERS, pin_memory=True)
    elif cfg.DATA.DATASET == 'aircraft':
        from .aircraft_2scale import AircraftDataset
        train_loader = torch.utils.data.DataLoader(
            AircraftDataset(cfg=cfg, is_train=True), batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=True, num_workers=cfg.BASIC.NUM_WORKERS, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            AircraftDataset(cfg=cfg, is_train=False), batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False, num_workers=cfg.BASIC.NUM_WORKERS, pin_memory=True)
    else:
        raise ValueError('Please set correct dataset.')

    print('done!')
    return train_loader, val_loader


def creat_data_loader_2scale_sr(cfg):
    print('==> Preparing data...')
    if cfg.DATA.DATASET == 'cub':
        from .cub_2scale_sr import CUBDataset
        train_loader = torch.utils.data.DataLoader(
            CUBDataset(cfg=cfg, is_train=True), batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=True, num_workers=cfg.BASIC.NUM_WORKERS, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            CUBDataset(cfg=cfg, is_train=False), batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False, num_workers=cfg.BASIC.NUM_WORKERS, pin_memory=True)
    else:
        raise ValueError('Please set correct dataset.')

    print('done!')
    return train_loader, val_loader

