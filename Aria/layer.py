import weakref
import numpy as np

import Aria.functions.Tensor as F

from Aria.core.Parameter import Parameter

class Layer:
  def __init__(self):
    self._params = set() # 매개변수

  def __setattr__(self, name, value):
    # 이름이 name인 인스턴스 변수에 값으로 value 전달(Parameter or Layer)
    if isinstance(value, (Parameter, Layer)):
      self._params.add(name)
    super().__setattr__(name, value)

  def __call__(self, *inputs):
    outputs = self.forward(*inputs) # 순전파 계산
    if not isinstance(outputs, tuple):
      outputs = (outputs,)
    self.inputs = [weakref.ref(x) for x in inputs] # 입력 값 저장
    self.outputs = [weakref.ref(y) for y in outputs] # 출력 값 저장
    return outputs if len(outputs) > 1 else outputs[0]
  
  def forward(self, inputs):
    raise NotImplementedError()
  
  def params(self):
    # Layer 인스턴스에 담긴 Parameter 인스턴스 반환
    for name in self._params:
      obj = self.__dict__[name]

      if isinstance(obj, Layer):
        yield from obj.params()
      else:
        yield obj
  
  def cleargrads(self):
    # 모든 매개변수의 미분값 재설정
    for param in self.params():
      param.cleargrad()

class Linear(Layer):
  def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
    super().__init__()
    self.in_size = in_size # 입력 크기
    self.out_size = out_size # 출력 크기
    self.dtype = dtype # 타입
    # in_size가 지정되지 않았다면 나중으로 연기
    self.W = Parameter(None, name='W')
    if self.in_size is not None:
      self._init_W()

    if nobias:
      self.b = None
    else:
      self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

  def _init_W(self):
    I, O = self.in_size, self.out_size
    self.W.data =  np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
  
  
  def forward(self, x):
    # 데이터를 보내는 시점에서 가중치 초기화
    if self.W.data is None:
      self.in_size = x.shape[1]
      self._init_W()

    return F.linear(x, self.W, self.b)