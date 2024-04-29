import numpy as np

from Aria.core.Config import Config
from Aria.core.Utils import as_variable

def dropout(x:np.ndarray, dropout_ratid:float=0.5) -> np.ndarray:
  """드롭아웃을 수행하는 함수

  Args:
    x (numpy.ndarray): 입력 배열
    dropout_ratid (Optional[float]): 드롭아웃 비율(기본값은 0.5)

  Returns:
    numpy.ndarray: 드롭아웃이 적용된 결과
  """
  x = as_variable(x)

  if Config.train:
    mask = np.random.rand(*x.shape) > dropout_ratid
    scale = np.array(1.0 - dropout_ratid).astype(x.dtype)
    return x * mask / scale
  else:
    return x