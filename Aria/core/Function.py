import weakref

import Aria

from Aria.core.Variable import Variable
from Aria.core.Config import Config

class Function:
  def __call__(self, *inputs):
    """객체가 호출될 때 함수를 호출하여 forward를 수행하는 메서드

    Args:
      *inputs (Tuple[numpy.ndarray | Variable]): 연산을 수행할 데이터들

    Returns:
      Union["Variable", Tuple["Variable"]]: forward 연산을 수행한 결과
    """
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
    """forward를 수행하는 메서드

    Args:
      x (numpy.ndarray): 입력 데이터

    Raises:
      NotImplementedError: 해당 메서드에서는 기능을 구현하지 않는다.

    Returns:
      numpy.ndarray: forward 연산을 수행한 결과
    """
    raise NotImplementedError()
  
  def backward(self, gy):
    """backward 수행하는 메서드

    Args:
      gy (numpy.ndarray): 입력 데이터

    Raises:
      NotImplementedError: 해당 메서드에서는 기능을 구현하지 않는다.
    """
    raise NotImplementedError()
