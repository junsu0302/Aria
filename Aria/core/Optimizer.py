class Optimizer:
  def __init__(self):
    """Optimizer 클래스의 생성자"""
    self.target = None # 매개변수를 갖는 클래스 (Model or Layer)
    self.hooks = [] # 전처리 함수

  def setup(self, target):
    """Optimizer의 대상 모델 설정

    Args:
      target (Model | Layer): 매개변수를 갖는 클래스

    Returns:
      Optimizer: 설정된 Opimizer 인스턴스 반환
    """
    self.target = target
    return self
  
  def update(self):
    """매개변수를 갱신하는 메서드"""
    params = [p for p in self.target.params() if p.grad is not None] # 매개변수 로드

    # 전처리
    for f in self.hooks:
      f(params)

    for param in params:
      self.update_one(param) # 매개변수 갱신

  def update_one(self, param):
    """하나의 매개변수를 갱신하는 메서드"""
    raise NotImplementedError()

  def add_hook(self, f):
    """전처리 함수를 추가하는 메서드

    Args:
      f (Function): 전처리 함수
    """
    self.hooks.append(f)