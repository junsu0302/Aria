from src.core.Variable import Variable

class Function:
  def __call__(self, input):
    x = input.data # 데이터 로드
    y = self.forward(x) # 순전파 계산
    output = Variable(y) # 계산 결과
    output.set_creator(self) # 부모 함수 설정
    self.input = input # 입력 값 저장
    self.output = output # 출력 값 저장
    return output
  
  def forward(self, x):
    raise NotImplementedError()
  
  def backward(self, gy):
    raise NotImplementedError()
