import numpy as np

class Dataset:
  def __init__(self, train=True, transform=None, target_transform=None):
    self.train = train
    self.transform = transform # 입력 데이터 전처리
    if self.transform is None:
      self.transform = lambda x: x
    self.target_transform = target_transform # 입력 데이터 레이블 전처리
    if self.target_transform is None:
      self.target_transform = lambda x: x

    self.data = None
    self.label = None
    self.prepare()

  def __getitem__(self, index):
    assert np.isscalar(index) # 스칼라만 지원
    if self.label is None:
      return self.transform(self.data[index]), None
    else:
      return self.transform(self.data[index]), self.target_transform(self.label[index])
    
  def __len__(self):
    return len(self.data)
  
  def prepare(self):
    pass