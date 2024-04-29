from typing import Union
import numpy as np

import Aria.activations as AF

from Aria.core.Layer import Layer
from Aria.core.Parameter import Parameter

from Aria.functions.Convolutional import conv2d
from Aria.functions.Tensor import linear
from Aria.functions.Basic import tanh

from Aria.functions.utils.Convolutional import pair

class Linear(Layer):
  """
  선형 변환을 수행하는 레이어
  
  - init(out_size, nobias, dtype, in_size)
  - forward(x) : 순전파 수행
  """
  def __init__(self, out_size:int, nobias:bool=False, dtype:np.dtype=np.float32, in_size:int=None) -> None:
    """선형 변환을 수행하는 레이어

    Args:
      out_size (int): 출력 크기
      nobias (bool, optional): 편향을 사용할지 여부(기본값은 False)
      dtype (numpy.dtype, optional): 데이터 타입(기본값은 numpy.float32)
      in_size (int, optional): 입력 크기(기본값은 None)
    """
    super().__init__()
    self.in_size = in_size # 입력 크기
    self.out_size = out_size # 출력 크기
    self.dtype = dtype # 타입
    # in_size가 지정되지 않았다면 나중으로 연기
    self.W = Parameter(None, name='W')
    if self.in_size is not None:
      self._init_W()

    if nobias:
      self.b = None
    else:
      self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

  def _init_W(self):
    """가중치 행렬 초기화"""
    I, O = self.in_size, self.out_size
    self.W.data =  np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
  
  def forward(self, x):
    """순전파 수행

    Args:
      x (Variable): 입력 데이터

    Returns:
      Variable: 출력 데이터
    """
    # 데이터를 보내는 시점에서 가중치 초기화
    if self.W.data is None:
      self.in_size = x.shape[1]
      self._init_W()

    return linear(x, self.W, self.b)
  
class Conv2d(Layer):
  """
  2차원 컨볼루션 레이어
  
  - init(out_channels, kernel_size, stride, pad, nobias, dtype, in_channels)
  - forward(x) : 순전파 수행
  """
  def __init__(self, out_channels:int, kernel_size:Union[int, tuple[int, int]], stride:Union[int, tuple[int, int]]=1, pad:Union[int, tuple[int, int]]=0, nobias:bool=False, dtype:np.dtype=np.float32, in_channels:int=None) -> None:
    """2차원 컨볼루션 레이어

    Args:
      out_channels (int): 출력 채널 수
      kernel_size (int | tuple[int, int]]): 커널의 크기
      stride (int | tuple[int, int]], optional): 스트라이드 값(기본값은 1)
      pad (int | tuple[int, int]], optional): 패딩의 크기(기본값은 0)
      nobias (bool, optional): 편향 사용 여부(기본값은 False)
      dtype (numpy.dtype, optional): 데이터 타입(기본값은 numpy.float32)
      in_channels (int, optional): 입력 채널 수(기본값은 None)
    """
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.pad = pad
    self.dtype = dtype

    self.W = Parameter(None, name='W')
    if in_channels is not None:
      self._init_W()

    if nobias:
      self.b = None
    else:
      self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')
    
  def _init_W(self, xp) -> None:
    """가중치 행렬 초기화"""
    C, OC = self.in_channels, self.out_channels
    KH, KW = pair(self.kernel_size)
    scale = np.sqrt(1 / (C * KH * KW))
    W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
    self.W.data = W_data

  def forward(self, x):
    """순전파 수행

    Args:
      x (Variable): 입력 데이터

    Returns:
      Variable: 출력 데이터
    """
    if self.W.data is None:
      self.in_channels = x.shape[1]
      self._init_W(x)

    return conv2d(x, self.W, self.b, self.stride, self.pad)
  
class RNN(Layer):
  """
  순환 신경망 레이어
  
  - init(self, hidden_size, in_size)
  - reset_state() : 은닉 상태 초기화
  - forward(x) : 순전파 수행
  """
  def __init__(self, hidden_size:int, in_size:int=None) -> None:
    """순환 신경망 레이어

    Args:
      hidden_size (int): 은닉 상태 크기
      in_size (int, optional): 입력 크기(기본값은 None)
    """
    super().__init__()
    self.x2h = Linear(hidden_size, in_size=in_size) # 입력에서 은닉 상태로 변환하는 완전연결계층
    self.h2h = Linear(hidden_size, in_size=in_size, nobias=True) # 이전 은닉 상태에서 다음 은닉 상태로 변환하는 완전연결계층
    self.h = None # 은닉 상태 유무

  def reset_state(self) -> None:
    """은닉 상태 초기화"""
    self.h = None

  def forward(self, x):
    """순전파 수행

    Args:
      x (Variable): 입력 데이터

    Returns:
      Variable: 출력 데이터
    """
    if self.h is None:
      h_new = tanh(self.x2h(x))
    else:
      h_new = tanh(self.x2h(x) + self.h2h(self.h))
    self.h = h_new
    return h_new
  
class LSTM(Layer):
  """
  LSTM 레이어
  
  - init(hidden_size, in_size)
  - reset_state(): 은닉 상태 초기화
  - forward(x): 순전파 수행
  """
  def __init__(self, hidden_size:int, in_size:int=None) -> None:
    """LSTM 생성자

    Args:
      hidden_size (int): 은닉 상태의 크기
      in_size (int, optional): 입력 크기(기본값은 None)
    """
    super().__init__()

    H, I = hidden_size, in_size
    self.x2f = Linear(H, in_size=I)
    self.x2i = Linear(H, in_size=I)
    self.x2o = Linear(H, in_size=I)
    self.x2u = Linear(H, in_size=I)
    self.h2f = Linear(H, in_size=H, nobias=True)
    self.h2i = Linear(H, in_size=H, nobias=True)
    self.h2o = Linear(H, in_size=H, nobias=True)
    self.h2u = Linear(H, in_size=H, nobias=True)
    self.reset_state()

  def reset_state(self):
    """은닉 상태 초기화"""
    self.h = None
    self.c = None

  def forward(self, x):
    """순전파 수행

    Args:
      x (Variable): 입력 데이터

    Returns:
      Variable: 출력 데이터
    """
    if self.h is None:
      f = AF.sigmoid(self.x2f(x))
      i = AF.sigmoid(self.x2i(x))
      o = AF.sigmoid(self.x2o(x))
      u = tanh(self.x2u(x))
    else:
      f = AF.sigmoid(self.x2f(x) + self.h2f(self.h))
      i = AF.sigmoid(self.x2i(x) + self.h2i(self.h))
      o = AF.sigmoid(self.x2o(x) + self.h2o(self.h))
      u = tanh(self.x2u(x) + self.h2u(self.h))

    if self.c is None:
      c_new = (i * u)
    else:
      c_new = (f * self.c) + (i * u)

    h_new = o * tanh(c_new)

    self.h, self.c = h_new, c_new
    return h_new