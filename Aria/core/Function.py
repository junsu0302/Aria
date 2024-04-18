import weakref

import Aria

from Aria.core.Variable import Variable
from Aria.core.Config import Config

class Function:
  def __call__(self, *inputs):
    inputs = [Aria.core.Utils.as_variable(x) for x in inputs] # 입력값 형변환

    xs = [x.data for x in inputs] # 데이터 로드
    ys = self.forward(*xs) # 순전파 계산
    if not isinstance(ys, tuple):
      ys = (ys,)
    outputs = [Variable(Aria.core.Utils.as_array(y)) for y in ys] # 계산 결과 형변환

    if Config.enable_backprop:
      self.generation = max([x.generation for x in inputs]) # 세대 설정
      for output in outputs:
        output.set_creator(self) # 부모 함수 설정
      self.inputs = inputs # 입력 값 저장
      self.outputs = [weakref.ref(output) for output in outputs] # 출력 값 저장
    
    return outputs if len(outputs) > 1 else outputs[0]
  
  def forward(self, x):
    raise NotImplementedError()
  
  def backward(self, gy):
    raise NotImplementedError()
