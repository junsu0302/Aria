class Optimizer:
  def __init__(self):
    self.target = None # 매개변수를 갖는 클래스 (Model or Layer)
    self.hooks = [] # 전처리 함수

  def setup(self, target):
    self.target = target
    return self
  
  def update(self):
    params = [p for p in self.target.params() if p.grad is not None] # 매개변수 로드

    # 전처리
    for f in self.hooks:
      f(params)

    for param in params:
      self.update_one(param) # 매개변수 갱신

  def update_one(self, param):
    raise NotImplementedError()

  def add_hook(self, f):
    self.hooks.append(f)