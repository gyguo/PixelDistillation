from .kd_loss import KDLoss
from .similarity_loss import SPLoss
from .at_loss import ATLoss
from .ickd_loss import ICKDLoss
from .dkd_loss import DKDLoss

__factory = {
    'kd': KDLoss,
    'sp': SPLoss,
    'at': ATLoss,
    'ickd': ICKDLoss,
    'dkd': DKDLoss,
}


def names():
  return sorted(__factory.keys())


def build_criterion(cfg):
  """
  Create a dataset instance.
  Parameters
  ----------
  name : str
    The dataset name. Can be one of __factory
  root : str
    The path to the dataset directory.
  """
  name = cfg.MODEL.KDTYPE
  if name not in __factory:
    raise NotImplementedError('The method does not have its own loss calculation method.')
  return __factory[name](cfg)
