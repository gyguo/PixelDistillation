"""
one stage
image sr from feature of input conv
"""
import os
import sys
sys.path.insert(0, './')
import datetime
import pprint
import argparse
from lib.core.utils import str_gpus, AverageMeter, accuracy, list2acc, save_checkpoint, map_sklearn
from lib.config.default import cfg_from_list, cfg_from_file, update_config
from lib.config.default import config as cfg
from lib.datasets import creat_data_loader_2scale
from lib.utils import mkdir, Logger, prepare_env_noseed
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')


def args_parser():
    parser = argparse.ArgumentParser(description='knowledge distillation')
    parser.add_argument('--config_file', type=str,
                        default='',
                        required=False, help='Optional config file for params')
    parser.add_argument('opts', help='see config.py for all options',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def creat_model_kd(cfg):
    print('==> Preparing networks for baseline...')
    # use gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str_gpus(cfg.BASIC.GPU_ID)
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"

    from lib.models.model_builder import build_model, ISRKDModelBuilder
    model_t = build_model(arch=cfg.MODEL.ARCH_T, num_classes=cfg.DATA.NUM_CLASSES, pretrained=cfg.MODEL.PRETRAIN_T)
    model_s = build_model(arch=cfg.MODEL.ARCH_S, num_classes=cfg.DATA.NUM_CLASSES, pretrained=cfg.MODEL.PRETRAIN_S)

    from lib.models.hrir import SR1x1
    crop_size_small = int(cfg.DATA.CROP_SIZE / cfg.DATA.LESSEN_RATIO)
    input_s = torch.ones(1, 3, crop_size_small, crop_size_small)
    _, feats_s = model_s(input_s, return_feat=True)
    model_sr = SR1x1(cfg, list(feats_s[cfg.FSR.POSITION].shape))

    model = ISRKDModelBuilder(model_s, model_t, model_sr, cfg.FSR.POSITION)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=cfg.SOLVER.START_LR, momentum=cfg.SOLVER.MUMENTUM,
                                weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.LR_STEPS,
                                                     gamma=cfg.SOLVER.LR_DECAY_FACTOR)
    if cfg.DATA.DATASET != 'imagenet':
        state_dict_t = torch.load(os.path.join(cfg.BASIC.ROOT_DIR, cfg.MODEL.MODELDICT_T))['state_dict']
        model.model_t.load_state_dict(state_dict_t)
    model = torch.nn.DataParallel(model).to(device)

    # loss
    cls_criterion = torch.nn.CrossEntropyLoss().to(device)
    from lib.losses import build_criterion
    from lib.losses.kd_loss import KDLoss
    pkd_criterion = KDLoss(cfg).to(device)
    isr_criterion = torch.nn.L1Loss().to(device)
    print('Preparing networks done!')
    return device, model, optimizer, scheduler, cls_criterion, pkd_criterion, isr_criterion


def main():
    # update parameters
    args = args_parser()
    update_config(args)

    # create checkpoint directory
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    cfg.BASIC.SAVE_DIR = os.path.join(
        cfg.BASIC.ROOT_DIR, 'ckpt', cfg.DATA.DATASET,
        '{}_{}'.format(cfg.MODEL.ARCH_S, int(cfg.DATA.CROP_SIZE / cfg.DATA.LESSEN_RATIO)),
        'isrd-p{}_{}_{}_{}_k{}_eta-{}_seed{}_{}_{}'.format(
        cfg.FSR.POSITION, cfg.MODEL.ARCH_T, cfg.MODEL.ARCH_S,
        cfg.DATA.CROP_SIZE, cfg.DATA.LESSEN_RATIO, cfg.FSR.ETA,
        cfg.BASIC.SEED, cfg.SOLVER.START_LR, cfg.BASIC.TIME))

    # prepare running environment for the whole project
    prepare_env_noseed(cfg)

    # start loging
    sys.stdout = Logger(cfg.BASIC.LOG_FILE)
    pprint.pprint(cfg)

    best_list = []
    for irun in range(1, 6):
        log_dir_irun = os.path.join(cfg.BASIC.LOG_DIR, str(irun))
        mkdir(log_dir_irun)
        logger_irun = SummaryWriter(log_dir_irun)

        ckpt_dir_irun = os.path.join(cfg.BASIC.CKPT_DIR, str(irun))
        mkdir(ckpt_dir_irun)

        device, model, optimizer, scheduler, cls_criterion, pkd_criterion, isr_criterion = creat_model_kd(cfg)
        train_loader, val_loader = creat_data_loader_2scale(cfg)

        best_acc = 0
        update_train_step = 0
        update_val_step = 0
        for epoch in range(1, cfg.SOLVER.NUM_EPOCHS+1):
            update_train_step = train_one_epoch(train_loader, model, device, cls_criterion, pkd_criterion, isr_criterion,
                                                optimizer, epoch, irun, logger_irun, cfg, update_train_step)
            scheduler.step()
            acc, update_val_step = val_one_epoch(val_loader, model, device, cls_criterion, epoch, irun,
                                                 logger_irun, cfg, update_val_step)

            # best accuracy and save checkpoint
            if acc > best_acc:
                best_acc = max(acc, best_acc)
                # torch.save({
                #     'epoch': epoch,
                #     'state_dict': model.module.model_s.state_dict(),
                #     'best_acc': best_acc,
                # }, os.path.join(ckpt_dir_irun, 'model_best.pth'))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                }, os.path.join(ckpt_dir_irun, 'model_best.pth'))

                print("Best epoch: {}".format(epoch))
            print("Best accuracy: {}".format(best_acc))
        best_list.append(best_acc)

        print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
        print(best_list)
        print('Mean: {}'.format(sum(best_list) / len(best_list)))


def train_one_epoch(train_loader, model, device, cls_criterion, pkd_criterion, isr_criterion, optimizer,
                    epoch, irun, logger, cfg, update_train_step):
    losses = AverageMeter()
    eval = AverageMeter()

    model.train()
    model.module.model_t.eval()
    for i, (input_large, input_small, target, image_large) in enumerate(train_loader):
        # update iteration steps
        update_train_step += 1

        target = target.to(device)
        input_large = input_large.to(device)
        input_small = input_small.to(device)

        cls_logits_s, image_sr, cls_logits_t= model(input_large, input_small, preReLU=True)
        kd_loss = pkd_criterion(cls_logits_s, cls_logits_t, target)
        isr_loss = isr_criterion(image_sr, input_large)
        loss = kd_loss + isr_loss*cfg.FSR.ETA

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        eval_res = accuracy(cls_logits_s.data, target, topk=(1,))[0]
        eval_res = eval_res.item()
        losses.update(loss.item(), input_large.size(0))
        eval.update(eval_res, input_large.size(0))
        logger.add_scalar('loss_iter/train', loss.item(), update_train_step)
        logger.add_scalar('eval_iter/train_eval', eval_res, update_train_step)

        if i % cfg.BASIC.DISP_FREQ == 0 or i == len(train_loader)-1:
            print(('Train Run [{0}] Epoch [{1}]: [{2}/{3}], lr: {lr:.5f} '
                   'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                   'ACC@1 {eval.val:.3f} ({eval.avg:.3f})'.format(
                irun, epoch, i+1, len(train_loader), loss=losses,
                eval=eval, lr=optimizer.param_groups[-1]['lr'])))

    return update_train_step


def val_one_epoch(val_loader, model, device, criterion, epoch, irun, logger, cfg, update_val_step):
    losses = AverageMeter()
    eval = AverageMeter()

    with torch.no_grad():
        model.eval()
        for i, (input, target, name) in enumerate(val_loader):
            # update iteration steps
            update_val_step += 1

            target = target.to(device)
            input = input.to(device)

            cls_logits = model(input)
            loss = criterion(cls_logits, target)

            eval_res = accuracy(cls_logits.data, target, topk=(1,))[0]
            eval_res = eval_res.item()
            losses.update(loss.item(), input.size(0))
            eval.update(eval_res, input.size(0))
            logger.add_scalar('loss_iter/val', loss.item(), update_val_step)
            logger.add_scalar('eval_iter/val_eval', eval_res, update_val_step)

            if i % cfg.BASIC.DISP_FREQ == 0 or i == len(val_loader)-1:
                print(('VAL Run [{0}] Epoch [{1}]: [{2}/{3}] '
                       'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                       'ACC@1 {eval.val:.3f} ({eval.avg:.3f})'.format(
                    irun, epoch, i+1, len(val_loader), loss=losses, eval=eval)))

        return eval.avg, update_val_step


if __name__ == "__main__":
    main()