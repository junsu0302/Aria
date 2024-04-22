import math
import random
import numpy as np

class DataLoader:
  def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
    self.dataset = dataset # 데이터셋
    self.batch_size = batch_size # 배치 크기
    self.shuffle = shuffle # 섞을지 여부
    self.data_size = len(dataset) # 데이터셋의 크기
    self.max_iter = math.ceil(self.data_size / batch_size) # 최대 반복 횟수
    self.gpu = gpu # GPU 사용 여부

    self.reset()

  def reset(self):
    self.iteration = 0
    if self.shuffle:
      self.index = np.random.permutation(len(self.dataset))
    else:
      self.index = np.arange(len(self.dataset))

  def __iter__(self):
    return self
  
  def __next__(self):
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
    return self.__next__()
  
  def to_cpu(self):
    self.gpu = False

  def to_gpu(self):
    self.gpu = True