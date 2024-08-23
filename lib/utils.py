import shutil
import os
import sys
import errno
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn


def rm(path):
  try:
    shutil.rmtree(path)
  except OSError as e:
    if e.errno != errno.ENOENT:
      raise


def mkdir(path):
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise


class Logger(object):
    def __init__(self,filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename,'a')

    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)


def fix_seed_all(cfg):
    # fix sedd
    fix_random_seed(cfg.BASIC.SEED)
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE


def backup_codes(root_dir, res_dir, backup_list):
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir) # delete
    os.makedirs(res_dir)
    for name in backup_list:
        shutil.copytree(os.path.join(root_dir, name), os.path.join(res_dir, name))
    print('codes backup at {}'.format(os.path.join(res_dir, name)))


def prepare_env_noseed(cfg):
    # backup codes
    if cfg.BASIC.BACKUP_CODES:
        backup_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'backup')
        rm(backup_dir)
        backup_codes(cfg.BASIC.ROOT_DIR, backup_dir, cfg.BASIC.BACKUP_LIST)

    # create save directory
    cfg.BASIC.CKPT_DIR = os.path.join(cfg.BASIC.SAVE_DIR, 'ckpt')
    mkdir(cfg.BASIC.CKPT_DIR)
    cfg.BASIC.LOG_DIR = os.path.join(cfg.BASIC.SAVE_DIR, 'log')
    mkdir(cfg.BASIC.LOG_DIR)
    cfg.BASIC.LOG_FILE = os.path.join(cfg.BASIC.SAVE_DIR, 'Log_' + cfg.BASIC.TIME + '.txt')

def prepare_env(cfg):
    # fix random seed
    fix_random_seed(cfg.BASIC.SEED)
    # cudnn
    cudnn.benchmark = cfg.CUDNN.BENCHMARK  # Benchmark will impove the speed
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC  #
    cudnn.enabled = cfg.CUDNN.ENABLE  # Enables benchmark mode in cudnn, to enable the inbuilt cudnn auto-tuner

    # backup codes
    if cfg.BASIC.BACKUP_CODES:
        backup_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'backup')
        rm(backup_dir)
        backup_codes(cfg.BASIC.ROOT_DIR, backup_dir, cfg.BASIC.BACKUP_LIST)

    # create save directory
    cfg.BASIC.CKPT_DIR = os.path.join(cfg.BASIC.SAVE_DIR, 'ckpt')
    mkdir(cfg.BASIC.CKPT_DIR)
    cfg.BASIC.LOG_DIR = os.path.join(cfg.BASIC.SAVE_DIR, 'log')
    mkdir(cfg.BASIC.LOG_DIR)
    cfg.BASIC.LOG_FILE = os.path.join(cfg.BASIC.SAVE_DIR, 'Log_' + cfg.BASIC.TIME + '.txt')