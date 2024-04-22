import numpy as np

from Aria.core.DataLoader import DataLoader

class SeqDataLoader(DataLoader):
  def __init__(self, dataset, batch_size, gpu=False):
    super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False, gpu=gpu)

  def __next__(self):
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