from typing import Optional, Union
import numpy as np

from Aria.core.Function import Function
from Aria.core.Utils import as_variable
from Aria.functions.utils.Convolutional import pair, get_conv_outsize, im2col
from Aria.functions.Tensor import linear
from Aria.functions.utils.Convolutional import im2col_array, col2im_array, get_deconv_outsize

class Conv2d(Function):
  """2차원 컨볼루션 연산의 순전파 및 역전파를 처리하는 클래스"""
  def __init__(self, stride:float=1, pad:float=0):
    """2차원 컨볼루션 연산 클래스 생성자

    Args:
      stride (float, optional): 컨볼루션 연산의 스트라이드(기본값은 1)
      pad (float, optional): 입력 주변에 추가할 패딩의 양(기본값은 0)
    """
    super().__init__()
    self.stride = pair(stride)
    self.pad = pair(pad)

  def forward(self, x:np.ndarray, W:np.ndarray, b:np.ndarray) -> np.ndarray:
    """컨볼루션 연산의 forward

    Args:
      x (numpy.ndarray): 입력 데이터
      W (numpy.ndarray): 컨볼루션 커널 가중치
      b (numpy.ndarray): 편향

    Returns:
      np.ndarray: 컨볼루션 연산 결과
    """
    KH, KW = W.shape[2:]
    col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)

    y = np.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
    if b is not None:
      y += b
    y = np.rollaxis(y, 3, 1)
    return y

  def backward(self, gy:np.ndarray) -> tuple[np.ndarray, np.ndarray, Union[np.ndarray,None]]:
    """컨볼루션 연산의 backward

    Args:
      gy (numpy.ndarray): 출력의 기울기

    Returns:
      tuple[np.ndarray, np.ndarray, np.ndarray|None]: 입력, 가중치, 편향의 기울기
    """
    x, W, b = self.inputs
    gx = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad, outsize=(x.shape[2], x.shape[3]))
    gW = Conv2DGradW(self)(x, gy)
    gb = None
    if b.data is not None:
      gb = gy.sum(axis=(0, 2, 3))
    return gx, gW, gb

def conv2d(x:np.ndarray, W:np.ndarray, b:np.ndarray=None, stride:float=1, pad:float=0) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """2차원 컨볼루션 연산 함수

  Args:
    x (numpy.ndarray): 입력 데이터
    W (numpy.ndarray): 컨볼루션 커널 가중치
    b (np.ndarray, optional): 편향(기본값은 None)
    stride (float, optional): 컨볼루션 연산의 스트라이드(기본값은 1)
    pad (float, optional): 입력 주변에 추가할 패딩의 양(기본값은 0)

  Returns:
    np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray|None]: 컨볼루션 연산 결과
  """
  return Conv2d(stride, pad)(x, W, b)

class Deconv2d(Function):
  """2차원 디컨볼루션 연산의 순전파 및 역전파를 처리하는 클래스"""
  def __init__(self, stride:float=1, pad:float=0, outsize:tuple[int, int]=None):
    """2차원 디컨볼루션 연산 클래스 생성자

    Args:
      stride (float, optional): 디컨볼루션 연산의 스트라이드(기본값은 1)
      pad (float, optional): 입력 주변에 추가할 패딩의 양(기본값은 0)
      outsize (tuple[int, int], optional): 출력의 크기(기본값은 None)
    """
    super().__init__()
    self.stride = pair(stride)
    self.pad = pair(pad)
    self.outsize = outsize

  def forward(self, x:np.ndarray, W:np.ndarray, b:np.ndarray) -> np.ndarray:
    """디컨볼루션 연산의 forward

    Args:
      x (numpy.ndarray): 입력 데이터
      W (numpy.ndarray): 디컨볼루션 커널 가중치
      b (numpy.ndarray): 편향

    Returns:
      np.ndarray: 디컨볼루션 연산 결과
    """
    Weight = W
    SH, SW = self.stride
    PH, PW = self.pad
    C, OC, KH, KW = Weight.shape
    N, C, H, W = x.shape
    if self.outsize is None:
      out_h = get_deconv_outsize(H, KH, SH, PH)
      out_w = get_deconv_outsize(W, KW, SW, PW)
    else:
      out_h, out_w = pair(self.outsize)
    img_shape = (N, OC, out_h, out_w)

    gcol = np.tensordot(Weight, x, (0, 1))
    gcol = np.rollaxis(gcol, 3)
    y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad, to_matrix=False)
    if b is not None:
      self.no_bias = True
      y += b.reshape((1, b.size, 1, 1))
    return y

  def backward(self, gy:np.ndarray) -> tuple[np.ndarray, np.ndarray, Union[np.ndarray,None]]:
    """디컨볼루션 연산의 backward

    Args:
      gy (numpy.ndarray): 출력의 기울기

    Returns:
      tuple[np.ndarray, np.ndarray, np.ndarray|None]: 입력, 가중치, 편향의 기울기
    """
    x, W, b = self.inputs

    gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
    f = Conv2DGradW(self)
    gW = f(gy, x)
    gb = None
    if b.data is not None:
      gb = gy.sum(axis=(0, 2, 3))
    return gx, gW, gb


def deconv2d(x:np.ndarray, W:np.ndarray, b:np.ndarray=None, stride:float=1, pad:float=0, outsize:tuple[int,int]=None) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray, Union[np.ndarray,None]]]:
  """2차원 디컨볼루션 연산 함수

  Args:
    x (numpy.ndarray): 입력 데이터
    W (numpy.ndarray): 디컨볼루션 커널 가중치
    b (numpy.ndarray): 편향(기본값은 None) 
    stride (float, optional): 디컨볼루션 연산의 스트라이드(기본값은 1)
    pad (float, optional): 입력 주변에 추가할 패딩의 양(기본값은 0)
    outsize (tuple[int, int], optional): 출력의 크기(기본값은 None) 
  
  Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray|None]): 디컨볼루션 연산의 결과
  """
  return Deconv2d(stride, pad, outsize)(x, W, b)


class Conv2DGradW(Function):
  """2차원 컨볼루션 가중치의 기울기를 계산의 순전파 및 역젼파를 처리하는 클래스"""
  def __init__(self, conv2d: Conv2d) -> None:
    """2차원 컨볼루션 가중치의 기울기를 계산하는 클래스 생성자

    Args:
      conv2d (Conv2d): 컨볼루션 계산 함수 객체
    """
    W = conv2d.inputs[1]
    kh, kw = W.shape[2:]
    self.kernel_size = (kh, kw)
    self.stride = conv2d.stride
    self.pad = conv2d.pad

  def forward(self, x:np.ndarray, gy:np.ndarray) -> np.ndarray:
    """컨볼루션 가중치의 기울기 계산

    Args:
      x (numpy.ndarray): 입력 데이터
      gy (numpy.ndarray): 출력의 크기

    Returns:
      np.ndarray: 컨볼루션 가중치의 기울기
    """
    col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
    gW = np.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
    return gW

  def backward(self, gys: tuple[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """역전파 수행

    Args:
      gys (tuple[np.ndarray]): 출력 기울기의 튜플

    Returns:
      tuple[np.ndarray, np.ndarray]: 입력의 기울기와 출력의 기울기
    """
    x, gy = self.inputs
    gW, = self.outputs

    xh, xw = x.shape[2:]
    gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad, outsize=(xh, xw))
    ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
    return gx, ggy

class Pooling(Function):
  """풀링 연산의 순전파 및 역전파를 처리하는 클래스"""
  def __init__(self, kernel_size:tuple[int, int], stride:float=1, pad:float=0) -> None:
    """Pooling 클래스의 생성자

    Args:
      kernel_size (tuple[int, int]): 풀링 커널의 크기
      stride (float, optional): 스트라이드(기본값은 1)
      pad (float, optional): 패딩의 크기(기본값은 0)
    """
    super().__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.pad = pad

  def forward(self, x:np.ndarray) -> np.ndarray:
    """풀링 연산의 forward

    Args:
      x (numpy.ndarray): 입력 데이터

    Returns:
      np.ndarray: 풀링 연산의 결과
    """
    col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)

    N, C, KH, KW, OH, OW = col.shape
    col = col.reshape(N, C, KH * KW, OH, OW)
    self.indexes = col.argmax(axis=2)
    return col.max(axis=2)

  def backward(self, gy:np.ndarray) -> np.ndarray:
    """풀링 연산의 backward

    Args:
      gy (numpy.ndarray): 출력 기울기

    Returns:
      np.ndarray: 입력 기울기
    """
    return Pooling2DGrad(self)(gy)


class Pooling2DGrad(Function):
  """풀링 연산의 역전파를 처리하는 클래스"""
  def __init__(self, mpool2d:Pooling) -> None:
    """Pooling2DGrad 클래스의 생성자

    Args:
        mpool2d (Pooling): Pooling 객체
    """
    self.mpool2d = mpool2d
    self.kernel_size = mpool2d.kernel_size
    self.stride = mpool2d.stride
    self.pad = mpool2d.pad
    self.input_shape = mpool2d.inputs[0].shape
    self.dtype = mpool2d.inputs[0].dtype
    self.indexes = mpool2d.indexes

  def forward(self, gy:np.ndarray) -> np.ndarray:
    """풀링 연산의 역전파 수행

    Args:
      gy (numpy.ndarray): 출력의 기울기

    Returns:
      np.ndarray: 입력의 기울기
    """
    N, C, OH, OW = gy.shape
    N, C, H, W = self.input_shape
    KH, KW = pair(self.kernel_size)

    gcol = np.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)

    indexes = (self.indexes.ravel() + np.arange(0, self.indexes.size * KH * KW, KH * KW))
        
    gcol[indexes] = gy.ravel()
    gcol = gcol.reshape(N, C, OH, OW, KH, KW)
    gcol = np.swapaxes(gcol, 2, 4)
    gcol = np.swapaxes(gcol, 3, 5)

    return col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride, self.pad, to_matrix=False)

  def backward(self, ggx:np.ndarray) -> np.ndarray:
    """풀링 연산의 역전파 수행

    Args:
      ggx (numpy.ndarray): 출력 기울기의 기울기

    Returns:
      np.ndarray: 입력 기울기의 기울기
    """
    f = Pooling2DWithIndexes(self.mpool2d)
    return f(ggx)


class Pooling2DWithIndexes(Function):
  """풀링 연산의 역전파의 역전파를 처리하는 클래스"""
  def __init__(self, mpool2d:Pooling) -> None:
    """Pooling2DWithIndexes 클래스의 생성자

    Args:
        mpool2d (Pooling): Pooling 객체
    """
    self.kernel_size = mpool2d.kernel_size
    self.stride = mpool2d.stride
    self.pad = mpool2d.pad
    self.input_shpae = mpool2d.inputs[0].shape
    self.dtype = mpool2d.inputs[0].dtype
    self.indexes = mpool2d.indexes

  def forward(self, x:np.ndarray) -> np.ndarray:
    """풀링 연산의 역전파의 역전파 수행

    Args:
      x (numpy.ndarray): 입력 데이터

    Returns:
      np.ndarray: 풀링 연산 결과
    """
    col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
    N, C, KH, KW, OH, OW = col.shape
    col = col.reshape(N, C, KH * KW, OH, OW)
    col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
    indexes = self.indexes.ravel()
    col = col[np.arange(len(indexes)), indexes]
    return col.reshape(N, C, OH, OW)

def pooling(x:np.ndarray, kernel_size:tuple[int,int], stride:float=1, pad:float=0) ->np.ndarray:
  """2차원 풀링 연산을 수행하는 함수

  Args:
    x (numpy.ndarray): 입력 데이터
    kernel_size (tuple[int,int]): 풀링 커널의 크기
    stride (float, optional): 스트라이드(기본값은 1)
    pad (float, optional): 패딩의 크기(기본값은 0)

  Returns:
    np.ndarray: 풀링 연산의 결과
  """
  return Pooling(kernel_size, stride, pad)(x)