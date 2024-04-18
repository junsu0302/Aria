import weakref

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