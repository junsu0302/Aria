from src.core.Utils import as_array
from src.core.Variable import Variable

class Function:
  def __call__(self, *inputs):
    xs = [x.data for x in inputs] # 데이터 로드
    ys = self.forward(*xs) # 순전파 계산
    if not isinstance(ys, tuple):
      ys = (ys,)
    outputs = [Variable(as_array(y)) for y in ys] # 계산 결과

    for output in outputs:
      output.set_creator(self) # 부모 함수 설정
    self.inputs = inputs # 입력 값 저장
    self.outputs = outputs # 출력 값 저장
    
    return outputs if len(outputs) > 1 else outputs[0]
  
  def forward(self, x):
    raise NotImplementedError()
  
  def backward(self, gy):
    raise NotImplementedError()
