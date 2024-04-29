from typing import Union
import numpy as np

from Aria.core.Function import Function

class Im2col(Function):
  """이미지를 입력으로 받아서 컬럼으로 변환하는 클래스"""
  def __init__(self, kernel_size:Union[int, tuple[int,int]], stride:Union[int, tuple[int,int]], pad:Union[int, tuple[int,int]], to_matrix:bool):
    """Im2col 클래스의 생성자

    Args:
        kernel_size (int | tuple[int,int]]): 커널의 크기
        stride (int | tuple[int,int]]): 스트라이드 값
        pad (int | tuple[int,int]]): 패딩의 크기
        to_matrix (bool): 행렬로 변환할지 결정하는 플래그
    """
    super().__init__()
    self.input_shape = None
    self.kernel_size = kernel_size
    self.stride = stride
    self.pad = pad
    self.to_matrix = to_matrix

  def forward(self, x:np.ndarray) -> np.ndarray:
    """Im2col의 forward

    Args:
      x (numpy.ndarray): 입력 이미지

    Returns:
      numpy.ndarray: 컬럼으로 변환된 이미지
    """
    self.input_shape = x.shape
    return im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)
    
  def backward(self, gy:np.ndarray) -> np.ndarray:
    """Im2col의 backward

    Args:
      gy (numpy.ndarray): 출력의 기울기 

    Returns:
      numpy.ndarray: 입력의 기울기
    """
    return col2im(gy, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)


def im2col(x:np.ndarray, kernel_size:Union[int, tuple[int,int]], stride:Union[int, tuple[int,int]]=1, pad:Union[int, tuple[int,int]]=0, to_matrix:bool=True) -> np.ndarray:
  """입력 이미지를 컬럼으로 변환하는 함수

  Args:
    x (numpy.ndarray): 입력 이미지
    kernel_size (int | tuple[int,int]): 커널의 크기
    stride (int | tuple[int,int], optional): 스트라이드 값(기본값은 1). Defaults to 1.
    pad (int | tuple[int,int], optional): 패딩의 크기(기본값은 0). Defaults to 0.
    to_matrix (bool, optional): 행렬로 변환할지 결정하는 플래그(기본값은 True). Defaults to True.

  Returns:
    numpy.ndarray: 컬럼으로 변환된 이미지
  """
  return Im2col(kernel_size, stride, pad, to_matrix)(x)

class Col2im(Function):
  """컬럼을 이미지로 변환하는 클래스"""
  def __init__(self, input_shape:tuple[int,int,int,int], kernel_size:Union[int, tuple[int,int]], stride:Union[int, tuple[int,int]], pad:Union[int, tuple[int,int]], to_matrix:bool) -> None:
    """Col2im 클래스의 생성자

    Args:
      input_shape (tuple[int,int,int,int]): 입력 이미지의 모양
      kernel_size (int | tuple[int,int]): 커널의 크기
      stride (int | tuple[int,int]): 스트라이드 값
      pad (int | tuple[int,int]): 패딩의 크기
      to_matrix (bool): 행렬로 변환할지 결정하는 플래그
    """
    super().__init__()
    self.input_shape = input_shape
    self.kernel_size = kernel_size
    self.stride = stride
    self.pad = pad
    self.to_matrix = to_matrix

  def forward(self, x:np.ndarray) -> np.ndarray:
    """Col2in의 forward

    Args:
      x (np.ndarray): 입력 컬럼

    Returns:
      np.ndarray: 복원된 이미지
    """
    return col2im_array(x, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)

  def backward(self, gy):
    """Col2im의 backward

    Args:
        gy (_type_): 출력의 기울기

    Returns:
        _type_: 입력의 기울기
    """
    return im2col(gy, self.kernel_size, self.stride, self.pad, self.to_matrix)

def col2im(x:np.ndarray, input_shape:tuple[int,int,int,int], kernel_size:Union[int, tuple[int,int]], stride:Union[int, tuple[int,int]]=1, pad:Union[int, tuple[int,int]]=0, to_matrix:bool=True) -> np.ndarray:
  """컬럼을 다시 이미지로 변환

  Args:
    x (np.ndarray): 입력 컬럼
    input_shape (tuple[int,int,int,int]): 입력 이미지의 모양
    kernel_size (int | tuple[int,int]): 커널의 크기
    stride (int | tuple[int,int], optional): 스트라이드 값(기본값은 1)
    pad (int | tuple[int,int], optional): 패딩의 크기(기본값은 0)
    to_matrix (bool, optional): 행렬로 변환할지 결정하는 플래그(기본값은 True)

  Returns:
    np.ndarray: 변환된 이미지
  """
  return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)

def im2col_array(img:np.ndarray, kernel_size:Union[int,tuple[int,int]], stride:Union[int,tuple[int,int]], pad: Union[int,tuple[int,int]], to_matrix:bool=True) -> np.ndarray:
  """입력 이미지를 컬럼으로 변환

  Args:
    img (numpy.ndarray): 입력 이미지
    kernel_size (int | tuple[int,int]]): 커널의 크기
    stride (int | tuple[int,int]]): 스프라이드 값
    pad (int | tuple[int,int]]): 패딩의크기
    to_matrix (bool, optional): 행렬로 변환힐지 결정하는 플래그(기본밗은 True)
      
  Returns:
    numpy.ndarray: 변환된 이미지
  """
  N, C, H, W = img.shape
  KH, KW = pair(kernel_size)
  SH, SW = pair(stride)
  PH, PW = pair(pad)
  OH = get_conv_outsize(H, KH, SH, PH)
  OW = get_conv_outsize(W, KW, SW, PW)

  img = np.pad(img, ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)), mode='constant', constant_values=(0,))
  col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

  for j in range(KH):
    j_lim = j + SH * OH
    for i in range(KW):
      i_lim = i + SW * OW
      col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

  if to_matrix:
    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

  return col

def col2im_array(col:np.ndarray, img_shape:tuple[int,int,int,int], kernel_size:Union[int,tuple[int,int]], stride:Union[int,tuple[int,int]], pad:Union[int,tuple[int,int]], to_matrix:bool=True) -> np.ndarray:
  """컬럼을 다시 이미지로 변환

  Args:
    col (numpy.ndarray): 입력 컬럼
    img_shape (tuple[int,int,int,int]): 입력 이미지 모양
    kernel_size (int | tuple[int,int]]): 커널의 크기
    stride (int | tuple[int,int]]): 스트라이드 값
    pad (int | tuple[int,int]]): 패딩의 크기
    to_matrix (bool, optional): 해렬로 변환할지 결정하는 플래그(기본값은 True)

  Returns:
    numpy.ndarray: 복원된 이미지
  """
  N, C, H, W = img_shape
  KH, KW = pair(kernel_size)
  SH, SW = pair(stride)
  PH, PW = pair(pad)
  OH = get_conv_outsize(H, KH, SH, PH)
  OW = get_conv_outsize(W, KW, SW, PW)

  if to_matrix:
    col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

  img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1), dtype=col.dtype)
  for j in range(KH):
    j_lim = j + SH * OH
    for i in range(KW):
      i_lim = i + SW * OW
      img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
  return img[:, :, PH:H + PH, PW:W + PW]

def get_deconv_outsize(size:int, k:int, s:int, p:int) -> int:
  """역합성곱 연산의 툴력 크기를 계산

  Args:
  size (int): 입력 크기
  k (int): 커널의 크기
  s (int): 스트라이드 값
  p (int): 패딩의 크기

  Returns:
  int: 출력 크기
  """
  return s * (size - 1) + k - 2 * p

def get_conv_outsize(input_size:int, kernel_size:Union[int,tuple[int,int]], stride:Union[int,tuple[int,int]], pad:Union[int,tuple[int,int]]) -> int:
  """합성곱 연산의 출력 크기를 계산

  Args:
    input_size (int): 입력 크기
    kernel_size (int | tuple[int,int]]): 커널의 크기
    stride (int | tuple[int,int]]): 스프라이드 값
    pad (int | tuple[int,int]]): 패딩의 크기

  Returns:
    int: 출력 크기
  """
  return (input_size + pad * 2 - kernel_size) // stride + 1

def pair(x:Union[int,tuple[int,int]]) -> tuple[int, int]:
  """입력 값을 쌍으로 반환

  Args:
    x (int | tuple[int,int]]): 입력 값

  Returns:
    tuple[int, int]: 입력 값이 정수일 경우에는 입력 값과 자기 자신의 튜플 반환, 튜플일 경우 그대로 반환
  """
  if isinstance(x, int):
    return (x, x)
  elif isinstance(x, tuple):
    assert len(x) == 2
    return x
  else:
    raise ValueError