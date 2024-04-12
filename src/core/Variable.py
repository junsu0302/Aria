import numpy as np

class Variable:
  def __init__(self, data):
    if data is not None:
      if not isinstance(data, np.ndarray):
        raise TypeError('\033[31m' + '{}은(는) 지원하지 않습니다.'.format(type(data)) + '\033[0m')

    self.data = data # 데이터
    self.grad = None # 기울기
    self.creator = None # 부모 함수
    self.generation = 0 # 세대 수

  def set_creator(self, func):
    self.creator = func
    self.generation = func.generation + 1

  def cleargrad(self):
    self.grad = None

  def backward(self, retain_grad=False):
    if self.grad is None:
      self.grad = np.ones_like(self.data)

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