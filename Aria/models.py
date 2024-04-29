from tkinter import Image
import numpy as np

from Aria.core.Model import Model
from Aria.utils.Visualize import plot_dot_graph
from Aria.utils.Download import get_file

import Aria.layers as L
import Aria.activations as AF
import Aria.functions.Convolutional as Conv
import Aria.functions.Tensor as F
import Aria.functions.Train as Train
  
class TwoLayerNet(Model):
  """
  두 개의 레이어로 이루어진 신경망 모델
  
  - init(hidden_size, out_size)
  - forward(x) : 순전파 수행
  """
  def __init__(self, hidden_size: int, out_size: int) -> None:
    """모두 개의 레이어로 이루어진 신경망 모델

    Args:
      hidden_size (int): 은닉층의 뉴런 수
      out_size (int): 출력층의 뉴런 수
    """
    super().__init__()
    self.l1 = L.Linear(hidden_size)
    self.l2 = L.Linear(out_size)

  def forward(self, x:np.ndarray) -> np.ndarray:
    """순전파 수행

    Args:
      x (numpy.ndarray): 입력 데이터

    Returns:
      numpy.ndarray: 출력 데이터
    """
    y = AF.sigmoid(self.l1(x))
    y = self.l2(y)
    return y
  
class MLP(Model):
  """
  다층 퍼셉트론 모델
  
  - init(fc_output_sizes, activation)
  - forward(x): 순전파 수행
  """
  def __init__(self, fc_output_sizes:list[int], activation:AF=AF.sigmoid) -> None:
    """다층 퍼셉트론 모델

    Args:
      fc_output_sizes (list[int]): 각 은닉층의 출력 크기 리스트
      activation (activations, optional): 활성화 함수(기본값은 Sigmoid)
    """
    super().__init__()
    self.activation = activation
    self.layers = []

    for i, out_size in enumerate(fc_output_sizes):
      layer = L.Linear(out_size)
      setattr(self, 'l' + str(i), layer)
      self.layers.append(layer)

  def forward(self, x:np.ndarray) -> np.ndarray:
    """순전파 수행

    Args:
      x (numpy.ndarray): 입력 데이터

    Returns:
      numpy.ndarray: 출력 데이턴
    """
    for l in self.layers[:-1]:
      x = self.activation(l(x))
    return self.layers[-1](x)
  
class VGG16(Model):
  """
  VGG16 모델
  
  - init(pretrained)
  """
  WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz'

  def __init__(self, pretrained:bool=False) -> None:
    """VGG16 모델

    Args:
      pretrained (bool, optional): 사전 학습된 가중치를 사용할지 여부 지정(기본값은 False)
    """
    super().__init__()
    self.conv1_1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
    self.conv1_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
    self.conv2_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.conv2_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.conv3_1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
    self.conv3_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
    self.conv3_3 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.conv4_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.conv4_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
    self.conv4_3 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
    self.conv5_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.conv5_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.conv5_3 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.fc6 = L.Linear(4096)
    self.fc7 = L.Linear(4096)
    self.fc8 = L.Linear(1000)

    if pretrained:
      self.load_weights('vgg16.npz')

  def forward(self, x:np.ndarray) -> np.ndarray:
    """순전파 수행

    Args:
        x (numpy.ndarray): 입력 데이터

    Returns:
        numpy.ndarray: 출력 데이터
    """
    x = AF.relu(self.conv1_1(x))
    x = AF.relu(self.conv1_2(x))
    x = Conv.pooling(x, 2, 2)
    x = AF.relu(self.conv2_1(x))
    x = AF.relu(self.conv2_2(x))
    x = Conv.pooling(x, 2, 2)
    x = AF.relu(self.conv3_1(x))
    x = AF.relu(self.conv3_2(x))
    x = AF.relu(self.conv3_3(x))
    x = Conv.pooling(x, 2, 2)
    x = AF.relu(self.conv4_1(x))
    x = AF.relu(self.conv4_2(x))
    x = AF.relu(self.conv4_3(x))
    x = Conv.pooling(x, 2, 2)
    x = AF.relu(self.conv5_1(x))
    x = AF.relu(self.conv5_2(x))
    x = AF.relu(self.conv5_3(x))
    x = Conv.pooling(x, 2, 2)
    x = F.reshape(x, (x.shape[0], -1))
    x = Train.dropout(AF.relu(self.fc6(x)))
    x = Train.dropout(AF.relu(self.fc7(x)))
    x = self.fc8(x)
    return x
  
  @staticmethod
  def preprocess(image: Image, size:tuple[int,int]=(224, 224), dtype:np.dtype=np.float32) -> np.ndarray:
    """이미지 전처리 수행

    Args:
      image (Image): 입력 이미지
      size (tuple[int,int], optional): 이미지 크기(기본값은 (224,224))
      dtype (numpy.dtype, optional): 데이터 타입(기본값은 numpy.float32)

    Returns:
      numpy.ndarray: 전처리된 이미지 데이터
    """
    image = image.convert('RGB')
    if size:
      image = image.resize(size)
    image = np.asarray(image, dtype=dtype)
    image = image[:, :, ::-1]
    image -= np.array([103.939, 116.779, 123.68], dtype=dtype)
    image = image.transpose((2, 0, 1))
    return image
  
class SimpleRNN(Model):
  """
  간단한 RNN 모델
  
  - init(hidden_size, out_size)
  - reset_state(): 상태 초기화
  - forward(x): 순전파 수행
  """
  def __init__(self, hidden_size:int, out_size:int) -> None:
    """간단한 RNN 모델

    Args:
      hidden_size (int): 은닉 상태의 크기
      out_size (int): 출력의 크기
    """
    super().__init__()
    self.rnn = L.RNN(hidden_size)
    self.fc = L.Linear(out_size)

  def reset_state(self):
    """상태를 초기화"""
    self.rnn.reset_state()
    
  def forward(self, x:np.ndarray) -> np.ndarray:
    """순전파 수행

    Args:
      x (numpy.ndarray): 입력 데이터

    Returns:
      numpy.ndarray: 출력 데이터
    """
    h = self.rnn(x)
    y = self.fc(h)
    return y
  
class SimpleLSTM(Model):
  """
  간단한 LSTM 모델

  - init(hidden_size, out_size)
  - reset_state(): 상태 초기화
  """
  def __init__(self, hidden_size:int, out_size:int) -> None:
    """간단한 LSTM 모델

    Args:
      hidden_size (int): 은닉 상태의 크기
      out_size (int): 출력의 크기
    """
    super().__init__()
    self.rnn = L.LSTM(hidden_size)
    self.fc = L.Linear(out_size)

  def reset_state(self):
    """상태 초기화"""
    self.rnn.reset_state()

  def __call__(self, x:np.ndarray) -> np.ndarray:
    """호출되었을 때의 동작 정의

    Args:
      x (numpy.ndarray): 입력 데이터

    Returns:
      numpy.ndarray: 출력 데이터
    """
    y = self.rnn(x)
    y = self.fc(y)
    return y