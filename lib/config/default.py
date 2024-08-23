import logging
import datetime
logger = logging.getLogger(__name__)


class AttrDict(dict):
    """
    Subclass dict and define getter-setter.
    This behaves as both dict and obj.
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


__C = AttrDict()
config = __C

__C.BASIC = AttrDict()
__C.BASIC.TIME = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
__C.BASIC.GPU_ID = [0]
__C.BASIC.NUM_WORKERS = 10
__C.BASIC.DISP_FREQ = 10  # frequency to display
__C.BASIC.SEED = 0
__C.BASIC.SAVE_DIR = ''
__C.BASIC.ROOT_DIR = ''
__C.BASIC.BACKUP_CODES = True
__C.BASIC.BACKUP_LIST = ['lib', 'tools']


# Model options
__C.MODEL = AttrDict()
__C.MODEL.TYPE = ''  # single, kd, pixel(ours)
__C.MODEL.ARCH = ''
__C.MODEL.PRETRAIN = False
__C.MODEL.KDTYPE = 'kd'  # kd, at
__C.MODEL.ARCH_T = ''  # teacher
__C.MODEL.MODELDICT_T = ''
__C.MODEL.PRETRAIN_T = False
__C.MODEL.ARCH_A = ''  # assistant
__C.MODEL.MODELDICT_A = ''
__C.MODEL.PRETRAIN_A = False
__C.MODEL.ARCH_S = ''  # student
__C.MODEL.MODELDICT_S = ''
__C.MODEL.PRETRAIN_S = False
__C.MODEL.PRERELU = False

__C.KD = AttrDict()
__C.KD.TEMP = 4
__C.KD.ALPHA = 0.9

__C.AT = AttrDict()
__C.AT.BETA = 1000.0

__C.SP = AttrDict()
__C.SP.BETA = 3000.0

__C.ICKD = AttrDict()
__C.ICKD.BETA = 2.5
__C.ICKD.FEATDIM_T = [512]
__C.ICKD.FEATDIM_S = [512]

__C.DKD = AttrDict()
__C.DKD.ALPHA = 1.0
__C.DKD.BETA = 8.0
__C.DKD.TEMP = 4
__C.DKD.WARMUP = 20

__C.FSR = AttrDict()
__C.FSR.BETA = 0.0
__C.FSR.GAMMA = 0.0
__C.FSR.ETA = 0.0
__C.FSR.BETA1 = 0.0
__C.FSR.BETA2 = 0.0
__C.FSR.RESIDUAL = False
__C.FSR.POSITION = 0


# Data options
__C.DATA = AttrDict()
__C.DATA.DATASET = ''
__C.DATA.DATADIR = ''
__C.DATA.NUM_CLASSES = 200
__C.DATA.RESIZE_SIZE = 256
__C.DATA.CROP_SIZE = 224
__C.DATA.LESSEN_RATIO = 1.0
__C.DATA.LESSEN_TYPE = 2  # 1 Nearest 2 Bilinear 3 Bicubic 4 Antialias
__C.DATA.IMAGE_MEAN = [0.485, 0.456, 0.406]
__C.DATA.IMAGE_STD = [0.229, 0.224, 0.225]

# solver options
__C.SOLVER = AttrDict()
__C.SOLVER.START_LR = 0.1
__C.SOLVER.LR_STEPS = [60, 120, 160]
__C.SOLVER.LR_DECAY_FACTOR = 0.1
__C.SOLVER.NUM_EPOCHS = 140
__C.SOLVER.WEIGHT_DECAY = 5e-4
__C.SOLVER.MUMENTUM = 0.9


# Training options.
__C.TRAIN = AttrDict()
__C.TRAIN.BATCH_SIZE = 32

# Testing options.
__C.TEST = AttrDict()
__C.TEST.BATCH_SIZE = 32

# Cudnn related setting
__C.CUDNN = AttrDict()
__C.CUDNN.BENCHMARK = False
__C.CUDNN.DETERMINISTIC = True
__C.CUDNN.ENABLE = True


def merge_dicts(dict_a, dict_b):
    from ast import literal_eval
    for key, value in dict_a.items():
        if key not in dict_b:
            raise KeyError('Invalid key in config file: {}'.format(key))
        if type(value) is dict:
            dict_a[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        # The types must match, too.
        old_type = type(dict_b[key])
        if old_type is not type(value) and value is not None:
                raise ValueError(
                    'Type mismatch ({} vs. {}) for config key: {}'.format(
                        type(dict_b[key]), type(value), key)
                )
        # Recursively merge dicts.
        if isinstance(value, AttrDict):
            try:
                merge_dicts(dict_a[key], dict_b[key])
            except BaseException:
                raise Exception('Error under config key: {}'.format(key))
        else:
            dict_b[key] = value


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen, Loader=yaml.FullLoader))
    merge_dicts(yaml_config, __C)


def cfg_from_list(args_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(args_list) % 2 == 0, 'Specify values or keys for args'
    for key, value in zip(args_list[0::2], args_list[1::2]):
        key_list = key.split('.')
        cfg = __C
        for subkey in key_list[:-1]:
            assert subkey in cfg, 'Config key {} not found'.format(subkey)
            cfg = cfg[subkey]
        subkey = key_list[-1]
        assert subkey in cfg, 'Config key {} not found'.format(subkey)
        try:
            # Handle the case when v is a string literal.
            val = literal_eval(value)
        except BaseException:
            val = value
        assert isinstance(val, type(cfg[subkey])) or cfg[subkey] is None, \
            'type {} does not match original type {}'.format(
                type(val), type(cfg[subkey]))
        cfg[subkey] = val


def update_config(args):
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    if args.opts is not None:
        cfg_from_list(args.opts)
