from src.core.Variable import Variable

class Function:
  def __call__(self, input):
    x = input.data # 데이터 로드
    y = self.forward(x) # 순전파 계산
    output = Variable(y) # 계산 결과
    return output
  
  def forward(self, x):
    raise NotImplementedError()
