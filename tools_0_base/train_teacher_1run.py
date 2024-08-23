import os
import sys
sys.path.insert(0, './')
import datetime
import pprint
import argparse
from lib.core.utils import str_gpus, AverageMeter, accuracy, list2acc, save_checkpoint, map_sklearn
from lib.config.default import update_config
from lib.config.default import config as cfg
from lib.datasets import creat_data_loader
from lib.utils import mkdir, Logger, prepare_env
import torch
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


def creat_model(cfg):
    print('==> Preparing networks for baseline...')
    # use gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str_gpus(cfg.BASIC.GPU_ID)
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"

    from lib.models.model_builder import build_model
    model = build_model(arch=cfg.MODEL.ARCH, num_classes=cfg.DATA.NUM_CLASSES, pretrained=cfg.MODEL.PRETRAIN)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=cfg.SOLVER.START_LR, momentum=cfg.SOLVER.MUMENTUM,
                                weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.LR_STEPS,
                                                     gamma=cfg.SOLVER.LR_DECAY_FACTOR)

    model = torch.nn.DataParallel(model).to(device)

    # loss
    criterion = torch.nn.CrossEntropyLoss().to(device)
    print('Preparing networks done!')
    return device, model, optimizer, scheduler, criterion


def main():
    # update parameters
    args = args_parser()
    update_config(args)

    # create checkpoint directory
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    cfg.BASIC.SAVE_DIR = os.path.join(cfg.BASIC.ROOT_DIR, 'ckpt', cfg.DATA.DATASET, '{}_{}_{}_seed{}_{}_{}'.format(
        cfg.MODEL.TYPE, cfg.MODEL.ARCH, cfg.DATA.CROP_SIZE, cfg.BASIC.SEED, cfg.SOLVER.START_LR, cfg.BASIC.TIME))
    # prepare running environment for the whole project
    prepare_env(cfg)

    # start loging
    sys.stdout = Logger(cfg.BASIC.LOG_FILE)
    pprint.pprint(cfg)
    logger = SummaryWriter(cfg.BASIC.LOG_DIR)

    device, model, optimizer, scheduler, criterion = creat_model(cfg)
    train_loader, val_loader = creat_data_loader(cfg)

    best_acc = 0
    update_train_step = 0
    update_val_step = 0
    for epoch in range(1, cfg.SOLVER.NUM_EPOCHS+1):
        update_train_step = train_one_epoch(train_loader, model, device, criterion, optimizer,
                                            epoch, logger, cfg, update_train_step)
        scheduler.step()
        acc, update_val_step = val_one_epoch(val_loader, model, device, criterion, epoch,
                                             logger, cfg, update_val_step)

        # remember best accuracy and save checkpoint
        if acc > best_acc:
            best_acc = max(acc, best_acc)
            torch.save({
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(cfg.BASIC.CKPT_DIR, 'model_best.pth'))
            print("Best epoch: {}".format(epoch))
        print("Best accuracy: {}".format(best_acc))
        print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))


def train_one_epoch(train_loader, model, device, criterion, optimizer, epoch, logger, cfg, update_train_step):
    losses = AverageMeter()
    eval = AverageMeter()

    model.train()
    for i, (input, target, name) in enumerate(train_loader):
        # update iteration steps
        update_train_step += 1

        target = target.to(device)
        input = input.to(device)

        cls_logits = model(input)
        loss = criterion(cls_logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        eval_res = accuracy(cls_logits.data, target, topk=(1,))[0]
        eval_res = eval_res.item()
        losses.update(loss.item(), input.size(0))
        eval.update(eval_res, input.size(0))
        logger.add_scalar('loss_iter/train', loss.item(), update_train_step)
        logger.add_scalar('eval_iter/train_eval', eval_res, update_train_step)

        if i % cfg.BASIC.DISP_FREQ == 0 or i == len(train_loader)-1:
            print(('Train Epoch: [{0}][{1}/{2}],lr: {lr:.5f}\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'ACC@1 {eval.val:.3f} ({eval.avg:.3f})'.format(
                epoch, i+1, len(train_loader), loss=losses,
                eval=eval, lr=optimizer.param_groups[-1]['lr'])))

    return update_train_step


def val_one_epoch(val_loader, model, device, criterion, epoch, logger, cfg, update_val_step):
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
                print(('VAL Epoch: [{0}][{1}/{2}]\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'ACC@1 {eval.val:.3f} ({eval.avg:.3f})'.format(
                    epoch, i+1, len(val_loader), loss=losses, eval=eval)))

        return eval.avg, update_val_step


if __name__ == "__main__":
    main()