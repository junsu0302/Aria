import numpy as np

from Aria.core.Function import Function
from Aria.core.Math import logsumexp

from Aria.activations import softmax

class MeanSquaredError(Function):
  """
  평균 제곱 오차 함수
  
  - forward(x0, x1): 순전파 수행
  - backward(gy): 역전파 수행
  """
  def forward(self, x0:np.ndarray, x1:np.ndarray) -> np.ndarray:
    """순전파 수행

    Args:
      x0 (numpy.ndarray): 입력 데이터
      x1 (numpy.ndarray): 타겟 데이터

    Returns:
      numpy.ndarray: 평균 제곱 오차
    """
    diff = x0 - x1
    return (diff ** 2).sum() / len(diff)
  
  def backward(self, gy:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """역전파 수행

    Args:
      gy (numpy.ndarray): 출력 기울기

    Returns:
      tuple[numpy.ndarray, numpy.ndarray]: 입력 기울기
    """
    x0, x1 = self.inputs
    diff = x0 - x1
    gx0 = gy * diff * (2. / len(diff))
    gx1 = -gx0
    return gx0, gx1
  
def mean_squared_error(x0:np.ndarray, x1:np.ndarray) -> np.ndarray:
  """평균 제곱 오차를 계산

  Args:
    x0 (np.ndarray): 예측값
    x1 (np.ndarray): 타겟값

  Returns:
    np.ndarray: 평균 제곱 오차값
  """
  return MeanSquaredError()(x0, x1)

class SoftmaxCrossEntropy(Function):
  """
  Softmax Cross Entropy 함수
  
  - forward(x, t): 순전파 수행
  - backward(gy): 역전파 수행
  """
  def forward(self, x:np.ndarray, t:np.ndarray) -> float:
    """순전파 수행

    Args:
      x (numpy.ndarray): 예측값
      t (numpy.ndarray): 타겟값

    Returns:
      float: 소프트맥스 크로스 앤트로피 값
    """
    N = x.shape[0]
    log_z = logsumexp(x, axis=1)
    log_p = x - log_z
    log_p = log_p[np.arange(N), t.ravel()]
    y = -log_p.sum() / np.float32(N)
    return y

  def backward(self, gy:np.ndarray) -> np.ndarray:
    """역전파 수행

    Args:
      gy (numpy.ndarray): 출력 기울기

    Returns:
      numpy.ndarray: 입력 기울기
    """
    x, t = self.inputs
    N, CLS_NUM = x.shape

    gy *= 1/N
    y = softmax(x)
    # convert to one-hot
    t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
    y = (y - t_onehot) * gy
    return y


def softmax_cross_entropy(x:np.ndarray, t:np.ndarray) -> float:
  """Softmax Cross Entropy 계산

  Args:
    x (numpy.ndarray): 예측값
    t (numpy.ndarray): 타겟값

  Returns:
    float: 소프트맥스 크로스 엔트로피 값
  """
  return SoftmaxCrossEntropy()(x, t)