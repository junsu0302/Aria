import numpy as np

from Aria.core.Config import Config
from Aria.core.Utils import as_variable

def dropout(x, dropout_ratid=0.5):
  x = as_variable(x)

  if Config.train:
    mask = np.random.rand(*x.shape) > dropout_ratid
    scale = np.array(1.0 - dropout_ratid).astype(x.dtype)
    return x * mask / scale
  else:
    return x