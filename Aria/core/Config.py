import contextlib

class Config:
  enable_backprop = True # 역전파 모드 활성화

@contextlib.contextmanager
def using_config(name, value):
  prev_value = getattr(Config, name)
  setattr(Config, name, value)
  try:
    yield
  finally:
    setattr(Config, name, prev_value)

def no_grad():
  return using_config('enable_backprop', False)

