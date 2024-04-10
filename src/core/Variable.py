import numpy as np

class Variable:
  def __init__(self, data):
    if data is not None:
      if not isinstance(data, np.ndarray):
        raise TypeError('\033[31m' + '{}은(는) 지원하지 않습니다.'.format(type(data)) + '\033[0m')

    self.data = data # 데이터
    self.grad = None # 기울기
    self.creator = None # 부모 함수

  def set_creator(self, func):
    self.creator = func

  def backward(self):
    if self.grad is None:
      self.grad = np.ones_like(self.data)

    funcs = [self.creator]
    while funcs:
      f = funcs.pop() # 함수 획득
      x, y = f.input, f.output # 함수 입출력 획득
      x.grad = f.backward(y.grad) # 역전파 진행

      if x.creator is not None:
        funcs.append(x.creator)