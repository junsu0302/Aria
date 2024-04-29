import numpy as np

class Compose:
  def __init__(self, transforms:list=[]):
    """여러 변환 함수를 조합하는 클래스

    Args:
      transforms (Optional[list]): 적용할 변환 함수 리스트(기본값은 [])
    """
    self.transforms = transforms

  def __call__(self, img:np.ndarray) -> np.ndarray:
    """입력 이미지에 연속적으로 변환 함수 적용

    Args:
      img (numpy.ndarray): 입력 이미지

    Returns:
      numpy.ndarray: 변환된 이미지
    """
    if not self.transforms:
      return img
    for t in self.transforms:
      img = t(img)
    return img
    
class Normalize:
  """입력값을 정규화"""
  def __init__(self, mean:float=0, std:float=1):
    """입력값을 정규화하는 클래스

    Args:
      mean (Optional[float]): 정규화 평균값(기본값은 0)
      std (Optional[float]): 정규화 표준편차(기본값은 0)
    """
    self.mean = mean
    self.std = std

  def __call__(self, array:np.ndarray) -> np.ndarray:
    """배열을 정규화

    Args:
      array (numpy.ndarray): 입력 배열

    Returns:
      numpy.ndarray: 정규화된 배열
    """
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

class Flatten:
  """다차원 배열을 1차원으로 변환"""
  def __call__(self, array:np.ndarray) -> np.ndarray:
    """다차원 배열을 1차원으로 변환

    Args:
        array (numpy.ndarray): 입력 배열

    Returns:
        numpy.ndarray: 1차원으로 변환된 배열
    """
    return array.flatten()
    
class AsType:
  """배열의 데이터 타입을 변경"""
  def __init__(self, dtype:type=np.float32):
    """배열의 데이터 타입을 변경하는 클래스

    Args:
      dtype (Optional[Any]): 변경할 데이터 타입(기본값은 numpy.float32)
    """
    self.dtype = dtype

  def __call__(self, array:np.ndarray) -> np.ndarray:
    """배열의 데이터 타입 변경

    Args:
      array (numpy.ndarray): 입력 배열

    Returns:
      numpy.ndarray: 데이터 타입이 변경된 배열
    """
    return array.astype(self.dtype)


ToFloat = AsType # AsType의 별칭으로 ToFloat를 사용