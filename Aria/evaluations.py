from Aria.core.Variable import Variable
from Aria.core.Utils import as_variable, as_array

def accuracy(y:Variable, t:Variable) -> Variable:
  """정확도 계산

  Args:
    y (Variable): 모델의 출력
    t (Variable): 타겟

  Returns:
    Variable: 정확도를 나타내는 변수
  """
  y, t = as_variable(y), as_variable(t)

  pred = y.data.argmax(axis=1).reshape(t.shape)
  result = (pred == t.data)
  acc = result.mean()
  return Variable(as_array(acc))