import numpy as np

from Aria.core.Function import Function

def as_array(x):
  """입력된 값을 NumPy 배열로 변환하는 함수

  Args:
    x (any): 변환할 값

  Returns:
    numpy.ndarray: 입력된 값을 NumPy 배열로 변환한 결과
  """
  import numpy as np

  if np.isscalar(x):
    return np.array(x)
  return x

def as_variable(obj):
  """입력된 객체를 Variable 인스턴스로 변환하는 함수

  Args:
    obj (any): 변환할 객체

  Returns:
    Variable: 입력된 객체를 Variable로 변환한 결과
  """
  from Aria.core.Variable import Variable
  
  if isinstance(obj, Variable):
    return obj
  return Variable(obj)

class GetItem(Function):
  def __init__(self, slices):
    """GetItem 클래스의 생성자

    Args:
      slices (Tuple[slice, ...]): 선택할 인덱스 슬라이스
    """
    self.slices = slices

  def forward(self, x):
    """forward를 수행하는 메서드

    Args:
      x (numpy.ndarray): 입력 데이터

    Returns:
      numpy.ndarray: 선택된 슬라이스에 해당하는 데이터
    """
    return x[self.slices]
  
  def backward(self, gy):
    """backward를 수행하는 메서드

    Args:
      gy (numpy.ndarray): 출력 쪽에서 전해지는 기울기
    """
    x, = self.inputs
    f = GetItemGrad(self.slices, x.shape)

class GetItemGrad(Function):
  def __init__(self, slices, in_shape):
    """GetItemGrad 클래스의 생성자

    Args:
      slices (Tuple[slice, ...]): 선택할 인덱스 슬라이스
      in_shape (Tuple[int, ...]): 입력 데이터의 형상
    """
    self.slices = slices
    self.in_shape = in_shape

  def forward(self, gy):
    """Forward를 수행하는 메서드

    Args:
      gy (numpy.ndarray): 출력 쪽에서 전해지는 기울기

    Returns:
      numpy.ndarray: 선택된 슬라이스에 해당하는 데이터의 기울기
    """
    gx = np.zeros(self.in_shape)
    np.add.at(gx, self.slices, gy)
    return gx
  
  def backward(self, ggx):
    """forward를 수행하는 메서드

    Args:
      ggx (numpy.ndarray): 출력 쪽에서 전해지는 상위 레이어의 기울기

    Returns:
      numpy.ndarray: 하위 레이어로 전돨되는 하위 기울기
    """
    return get_item(ggx, self.slices)
  
def get_item(x, slices):
  """입력 데이터에 인덱싱을 수행하는 함수

  Args:
    x (numpy.ndarray): 입력 데이터
    slices (Tuple[slice, ...]): 선택할 인덱스 슬라이스

  Returns:
    numpy.ndarray: 선택된 슬라이스에 해당하는 데이터
  """
  f = GetItem(slices)
  return f(x)
