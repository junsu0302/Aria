import numpy as np

import Aria

from Aria.core.Config import using_config

class Variable:
  """Tensor 변수를 표현하는 클래스"""
  __array_priority__ = 200 # 인스턴스 연산자 우선순위 부여

  def __init__(self, data, name=None):
    """Tensor 변수를 표현하는 클래스

    Args:
      data (numpy.ndarray): Variable에 저장될 데이터
      name (Optional[str]): Variable의 이름(기본값은 None)

    Raises:
      TypeError: data가 None이 아니고, numpy.ndarray 형식이 아닌 경우 발생
    """
    if data is not None:
      if not isinstance(data, np.ndarray):
        raise TypeError('\033[31m' + '{}은(는) 지원하지 않습니다.'.format(type(data)) + '\033[0m')

    self.name = name # 이름
    self.data = data # 데이터
    self.grad = None # 기울기
    self.creator = None # 부모 함수
    self.generation = 0 # 세대 수

  @property
  def shape(self):
    """Variable의 형태를 반환하는 속성

    Returns:
      Tuple[int, ...]: Variable의 형태를 나타내는 튜플
    """
    return self.data.shape
  
  @property
  def ndim(self):
    """Variable의 차원 수를 반환하는 속성

    Returns:
      int: Variable의 차원 수
    """
    return self.data.ndim
  
  @property
  def size(self):
    """Variable의 요소 수를 반환하는 속성

    Returns:
      int: Variable의 요소 수
    """
    return self.data.size
  
  @property
  def dtype(self):
    """Variable의 데이터 타입을 반환하는 속성

    Returns:
      Type[numpy.dtype]: Variable의 데이터 타입
    """
    return self.data.dtype
  
  def reshape(self, *shape):
    """다차원 배열의 형태를 변환하는 메서드

    Args:
      *shape (Union[int, Tuple[int, ...]]): 형태를 변환할 다차원 배열의 차원을 나타내는 정수들

    Returns:
      Variable: 형태가 변환된 다차원 배열을 나타내는 Variable 객체
        """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
      shape = shape[0]
    return Aria.functions.Tensor.reshape(self, shape)
  
  def transpose(self, *axes):
    """다차원 배열의 축을 변경하는 메서드

    Args:
      *axes (Union[None, int, Tuple[int, ...]]): 변경할 축을 나타내는 정수들

    Returns:
      Variable: 축이 변경된 다차원 배열을 나다내는 Variable 객체
    """
    if len(axes) == 0:
      axes = None
    elif len(axes) == 1:
      if isinstance(axes[0], (tuple, list)) or axes[0] is None:
        axes = axes[0]
    return Aria.functions.Tensor.transpose(self, axes)
  
  @property
  def T(self):
    """다차원 배열의 전치를 반환하는 속성

    Returns:
      Variable: 전치된 다차원 배열을 나타내는 Variable 객체
    """
    return Aria.functions.Tensor.transpose(self)
  
  def sum(self, axis=None, keepdims=False):
    """다차원 배열의 합을 계산하는 메서드

    Args:
      axis (Optional[int | Tuple[int]]): 합을 계산할 축을 나하태는 정수(기본값은 None)
      keepdims (Optional[bool]): 출력에서 입력 차원 형태를 유지할지 여부를 나타내는 플래그(기본값은 None)

    Returns:
      Variable: 합이 계산된 다차원 배열을 나타내는 Variable 객체
    """
    return Aria.functions.Tensor.sum(self, axis, keepdims)
  
  def __len__(self):
    """다차원 배열의 첫 번째 차원의 길이를 반환하는 메서드

    Returns:
      int: 다차원 배열의 첫 번째 차원의 길이
    """
    return len(self.data)
  
  def __repr__(self):
    """다차원 배열을 문자열로 출력하는 메서드

    Returns:
      str: 다차원 배열의 문자열 표현
    """
    if self.data is None:
      return 'None'
    return str(self.data)

  def set_creator(self, func):
    """Variable의 creator를 설정하는 메서드

    Args:
      func (Function): Variable을 생성하는 생성자 함수
    """
    self.creator = func
    self.generation = func.generation + 1

  def cleargrad(self):
    """Variable의 grad를 초기화하는 메서드"""
    self.grad = None

  def unchain(self):
    """Variable와 creator 함수를 분리하는 메서드"""
    self.creator = None

  def backward(self, retain_grad=False, create_graph=False):
    """Variable에 대한 backward를 수행하는 메서드

    Args:
      retain_grad (Optional[bool]): 중간 미분값을 보존할지 여부를 나타내는 플래그(기본값은 False)
      create_graph (Optional[bool]): 역전파 그래프를 생성할지 여부를 나타내는 플래그(기본값은 False)
    """
    if self.grad is None:
      self.grad = Variable(np.ones_like(self.data))

    funcs = []
    seen_set = set()

    def add_func(f):
      """함수를 funcs 리스트에 추가(세대를 기준으로 정렬)"""
      if f not in seen_set:
        funcs.append(f)
        seen_set.add(f)
        funcs.sort(key=lambda x: x.generation)

    add_func(self.creator)

    while funcs:
      f = funcs.pop() # 함수 획득
      gys = [output().grad for output in f.outputs] # 미분값 획득

      with using_config('enable_backprop', create_graph): # 역전파 활성화 모드
        gxs = f.backward(*gys) # 역전파 호출
        if not isinstance(gxs, tuple):
          gxs = (gxs,)

        for x, gx in zip(f.inputs, gxs):
          # 역전파 결과 저장
          if x.grad is None:
            x.grad = gx
          else:
            x.grad = x.grad + gx

          if x.creator is not None:
            add_func(x.creator)

        if not retain_grad:
          for y in f.outputs:
            y().grad = None # 중간 미분값 삭제

  def unchain_backward(self):
    """Varialbe과 연결된 함수들의 연결을 해제하는 메서드"""
    if self.creator is not None:
      funcs = [self.creator]
      while funcs:
        f = funcs.pop()
        for x in f.inputs:
          if x.creator is not None:
            funcs.append(x.creator)
            x.unchain()

def setup_variable():
  """Variable 클래스에 다양한 연산을 설정하는 함수"""
  from Aria.core.Math import add, sub, rsub, mul, div, rdiv, neg, pow
  from Aria.functions.Tensor import matmul, max, min
  from Aria.core.Utils import get_item

  Variable.__add__ = add
  Variable.__radd__ = add
  Variable.__sub__ = sub
  Variable.__rsub__ = rsub
  Variable.__mul__ = mul
  Variable.__rmul__ = mul
  Variable.__truediv__ = div
  Variable.__rtruediv__ = rdiv
  Variable.__neg__ = neg
  Variable.__pow__ = pow

  Variable.matmaul = matmul
  Variable.dot = matmul
  Variable.max = max
  Variable.min = min

  Variable.__getitem__ = get_item