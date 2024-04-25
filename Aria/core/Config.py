import contextlib

class Config:
  enable_backprop = True # 역전파 모드 활성화
  train = True # 학습 모드 활성화

@contextlib.contextmanager
def using_config(name: str, value: bool):
  """지정된 Config 속성을 변경하여 임시로 설정 값을 적용하는 context manager

  Args:
    name (srt): 변경할 Config 속성 이름
    value (bool): 설정할 값
  """
  prev_value = getattr(Config, name)
  setattr(Config, name, value)
  try:
    yield
  finally:
    setattr(Config, name, prev_value)

def no_grad():
  """backward를 비활성화하는 no_grad context manager 반환

  Returns:
    ContextManager: enable_backprop을 False로 설정하는 context manager
  """
  return using_config('enable_backprop', False)

def test_mode():
  """테스트 모드를 활성화하는 test_mode context manager 반환

  Returns:
    ContextManager: train을 False로 설정하는 context manager
  """
  return using_config('train', False)
