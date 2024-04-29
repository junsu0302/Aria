import numpy as np

class Normalize:
  """주어진 배열 정규화"""
  def __init__(self, mean:float=0, std:float=1) -> None:
    """주어진 배열 정규화

    Args:
      mean (float, optional): 평균(기본값은 0)
      std (float, optional): 표준편차(기본값은 1)
    """
    self.mean = mean
    self.std = std

  def __call__(self, array:np.ndarray) -> np.ndarray:
    """주어진 배열을 정규화된 배열로 변환"""
    mean, std = self.mean, self.std

    if not np.isscalar(mean):
      mshape = [1] * array.ndim
      mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
      mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
    if not np.isscalar(std):
      rshape = [1] * array.ndim
      rshape[0] = len(array) if len(self.std) == 1 else len(self.std)
      std = np.array(self.std, dtype=array.dtype).reshape(*rshape)
    return (array - mean) / std
  
class Compose:
  def __init__(self, transforms=[]):
    self.transforms = transforms

  def __call__(self, img):
    if not self.transforms:
      return img
    for t in self.transforms:
      img = t(img)
    return img