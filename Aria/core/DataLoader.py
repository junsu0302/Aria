import math
import random
import numpy as np

class DataLoader:
  def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
    """DataLoader 클래스의 생성자

    Args:
      dataset (callable): DataLoader에 사용될 Dataset 인스턴스
      batch_size (int): 배치 크기
      shuffle (Optional[bool]): 데이터를 섞을지 여부를 지정하는 플래그(기본값은 True)
      gpu (Optional[bool]): GPU를 사용할지 여부를 지정하는 플래그(기본값은 False)
    """
    self.dataset = dataset # 데이터셋
    self.batch_size = batch_size # 배치 크기
    self.shuffle = shuffle # 섞을지 여부
    self.data_size = len(dataset) # 데이터셋의 크기
    self.max_iter = math.ceil(self.data_size / batch_size) # 최대 반복 횟수
    self.gpu = gpu # GPU 사용 여부

    self.reset()

  def reset(self):
    """DataLoader의 상태를 초기화하는 메서드"""
    self.iteration = 0
    if self.shuffle:
      self.index = np.random.permutation(len(self.dataset))
    else:
      self.index = np.arange(len(self.dataset))

  def __iter__(self):
    """DataLoader 객체를 반복자 설정"""
    return self
  
  def __next__(self):
    """다음 배치 데이터를 반환하는 메서드

    Returns:
      tuple: 입력 데이터 배치와 타겟 데이터 배치로 구성된 튜플 반환
    """
    if self.iteration >= self.max_iter:
      self.reset()
      raise StopIteration
    
    i, batch_size = self.iteration, self.batch_size
    batch_index = self.index[i * batch_size:(i+1) * batch_size]
    batch = [self.dataset[i] for i in batch_index]
    x = np.array([example[0] for example in batch])
    t = np.array([example[1] for example in batch])

    self.iteration += 1
    return x, t
  
  def next(self):
    """다음 배치 데이터를 반환하는 메서드

    Returns:
      tuple: 입력 데이터 배치와 타겟 데이터 배치로 구성된 튜플 반환
    """
    return self.__next__()
  
  def to_cpu(self):
    """DataLoader의 GPU 사용 여부를 CPU로 전환"""
    self.gpu = False

  def to_gpu(self):
    """DataLoader의 CPU 사용 여부를 GPU로 전환"""
    self.gpu = True