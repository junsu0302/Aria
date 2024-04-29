import numpy as np

class Dataset:
  def __init__(self, train=True, transform=None, target_transform=None):
    """Dataset 클래스의 생성자

    Args:
      train (Optional[bool]): 훈련 데이터인지 테스트 데이터인지 결정하는 플래그(기본값은 True)
      transform (Optional[callable]): 입력 데이터의 전처리 함수(기본값은 None)
      target_transform (Optional[callable]): 입력 데이터 레이블 전처리 함수. (기본값은 None)
    """
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
    """지정된 인덱스에 해당하는 데이터 샘플을 반환하는 메서드

    Args:
      index (int): 데이터 샘플의 인덱스

    Returns:
      tuple: 전처리된 입력 데이터와 레이블 데이터의 쌍
    """
    assert np.isscalar(index) # 스칼라만 지원
    if self.label is None:
      return self.transform(self.data[index]), None
    else:
      return self.transform(self.data[index]), self.target_transform(self.label[index])
    
  def __len__(self):
    """데이터셋의 길이를 반환하는 메서드

    Returns:
      int: 데이터셋의 샘플 수
    """
    return len(self.data)
  
  def prepare(self):
    """데이터셋을 준비하는 메서드"""
    pass