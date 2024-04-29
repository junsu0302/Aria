import numpy as np

from Aria.core.DataLoader import DataLoader

class SeqDataLoader(DataLoader):
  """
  순차적인 데이터를 로드하는 DataLoader
  
  - init(self, dataset, batch_size, gpu)
  """
  def __init__(self, dataset, batch_size:int, gpu:bool=False):
    """순차적인 데이터를 로드하는 DataLoader

    Args:
      dataset (Dataset): 로드할 데이터셋
      batch_size (int): 배치 크기
      gpu (bool, optional): GPU 사용 여부(기본값은 False)
    """
    super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False, gpu=gpu)

  def __next__(self) -> tuple[np.ndarray,np.ndarray]:
    """다음 배치 반환

    Returns:
      tuple[numpy.ndarray,numpy.ndarray]: 입력과 타겟의 배치
    """
    if self.iteration >= self.max_iter:
      self.reset()
      raise StopIteration

    jump = self.data_size // self.batch_size
    batch_index = [(i * jump + self.iteration) % self.data_size for i in range(self.batch_size)]
    batch = [self.dataset[i] for i in batch_index]

    x = np.array([example[0] for example in batch])
    t = np.array([example[1] for example in batch])

    self.iteration += 1
    return x, t