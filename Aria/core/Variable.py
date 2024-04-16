import numpy as np

import Aria

from Aria.core.Config import using_config

class Variable:
  __array_priority__ = 200 # 인스턴스 연산자 우선순위 부여

  def __init__(self, data, name=None):
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
    return self.data.shape
  
  @property
  def ndim(self):
    return self.data.ndim
  
  @property
  def size(self):
    return self.data.size
  
  @property
  def dtype(self):
    return self.data.dtype
  
  def __len__(self):
    return len(self.data)
  
  def __repr__(self):
    if self.data is None:
      return 'None'
    return str(self.data)

  def reshape(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
      shape = shape[0]
    return Aria.functions.Tensor.reshape(self, shape)
  
  def transpose(self):
    return Aria.functions.Tensor.transpose(self)
  
  @property
  def T(self):
    return Aria.functions.Tensor.transpose(self)
  
  def sum(self, axis=None, keepdims=False):
    return Aria.functions.Tensor.sum(self, axis, keepdims)

  def set_creator(self, func):
    self.creator = func
    self.generation = func.generation + 1

  def cleargrad(self):
    self.grad = None

  def backward(self, retain_grad=False, create_graph=False):
    """
    retain_grad : 중간 미분값 저장 모드
    create_graph : 역전파 활성화 모드
    """
    if self.grad is None:
      self.grad = Variable(np.ones_like(self.data))

    funcs = []
    seen_set = set()

    def add_func(f):
      # 함수 세대에 맞게 정렬하며 저장
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

def setup_variable():
  from Aria.core.Math import add, sub, rsub, mul, div, rdiv, neg, pow

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