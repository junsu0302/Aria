import gzip
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

from Aria.core.Dataset import Dataset
from Aria.functions.Transforms import Compose, Flatten, ToFloat, Normalize
from Aria.utils.Download import get_file

class Spiral(Dataset):
  """
  나선형 데이터셋
  
  - prepare() : 데이터셋 준비
  """
  def prepare(self) -> None:
    """데이터셋 준비"""
    self.data, self.label = get_spiral(self.train)

def get_spiral(train:bool=True) -> tuple[np.ndarray,np.ndarray]:
  """나선형 데이터 생성

  Args:
    train (bool, optional): 훈련 데이터를 생성할지 여부(기본값은 True)

  Returns:
    tuple[numpy.ndarray,numpy.ndarray]: 입력 데이터와 타겟 데이터로 이루어진 튜플
  """
  seed = 1984 if train else 2020
  np.random.seed(seed=seed)

  num_data, num_class, input_dim = 100, 3, 2
  data_size = num_class * num_data
  x = np.zeros((data_size, input_dim), dtype=np.float32)
  t = np.zeros(data_size, dtype=np.int32)

  for j in range(num_class):
    for i in range(num_data):
      rate = i / num_data
      radius = 1.0 * rate
      theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
      ix = num_data * j + i
      x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
      t[ix] = j

  indices = np.random.permutation(num_data * num_class)
  x = x[indices]
  t = t[indices]
  return x, t

class MNIST(Dataset):
  """
  MNIST 데이터셋

  - init(self, train, transform, target_transform)
  - prepare(): 데이터셋 준비
  - show(row, col): MNIST 이미지 반환
  """
  def __init__(self, train:bool=True, transform:np.ndarray=Compose([Flatten(), ToFloat(), Normalize(0., 255.)]), target_transform:Callable=None) -> None:
    """MNIST 데이터셋의 생성자

    Args:
      train (bool, optional): 훈련 데이터셋인지 테스트 데이터셋인지 지정(기본값은 True)
      transform (numpy.ndarray, optional): 입력 데이터에 적용할 변환을 지정(기본값은 Compose([Flatten(), ToFloat(), Normalize(0., 255.)]))
      target_transform (Callable, optional): 타겟 데이터에 적용할 변환을 지정(기본값은 None)
    """
    super().__init__(train, transform, target_transform)

  def prepare(self) -> None:
    """데이터셋 준비"""
    url = 'http://yann.lecun.com/exdb/mnist/'
    train_files = {'target': 'train-images-idx3-ubyte.gz', 'label': 'train-labels-idx1-ubyte.gz'}
    test_files = {'target': 't10k-images-idx3-ubyte.gz', 'label': 't10k-labels-idx1-ubyte.gz'}

    files = train_files if self.train else test_files
    data_path = get_file(url + files['target'])
    label_path = get_file(url + files['label'])

    self.data = self._load_data(data_path)
    self.label = self._load_label(label_path)

  def _load_label(self, filepath) -> np.ndarray:
    """라벨 데이터 로드"""
    with gzip.open(filepath, 'rb') as f:
      labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels

  def _load_data(self, filepath) -> np.ndarray:
    """이미지 데이터 로드"""
    with gzip.open(filepath, 'rb') as f:
      data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 1, 28, 28)
    return data

  def show(self, row:int=10, col:int=10) -> None:
    """MNIST 이미지 반환

    Args:
      row (int, optional): row(기본값은 10)
      col (int, optional): col(기본값은 10)
    """
    H, W = 28, 28
    img = np.zeros((H * row, W * col))
    for r in range(row):
      for c in range(col):
        img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[np.random.randint(0, len(self.data) - 1)].reshape(H, W)
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.show()

  @staticmethod
  def labels() -> dict[int, str]:
    """MNIST 클래스의 라벨 반환"""
    return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
  
class ImageNet(Dataset):
  """ImageNet 데이터셋"""
  def __init__(self) -> None:
    """ImageNet의 생성자"""
    NotImplemented

  @staticmethod
  def labels() -> dict[int, str]:
    """ImageNet 클래스의 라벨 반환"""
    url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
    path = get_file(url)
    with open(path, 'r') as f:
      labels = eval(f.read())
    return labels
  
class SinCurve(Dataset):
  """
  사인 곡선 데이터셋
  
  - prepare(): 데이터셋 준비
  """
  def prepare(self) -> None:
    """데이터셋 준비"""
    num_data = 1000
    dtype = np.float64

    x = np.linspace(0, 2 * np.pi, num_data)
    noise_range = (-0.05, 0.05)
    noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
    if self.train:
      y = np.sin(x) + noise
    else:
      y = np.cos(x)
    y = y.astype(dtype)
    self.data = y[:-1][:, np.newaxis]
    self.label = y[1:][:, np.newaxis]