import weakref
import os
import numpy as np

from Aria.core.Parameter import Parameter

class Layer:
  def __init__(self):
    """_summary_
    """
    self._params = set() # 매개변수

  def __setattr__(self, name, value):
    """인스턴스 변수에 값을 할당하는 메서드

    Args:
      name (str): 변수 이름
      value (Any): 할당할 값
    """
    # 이름이 name인 인스턴스 변수에 값으로 value 전달(Parameter or Layer)
    if isinstance(value, (Parameter, Layer)):
      self._params.add(name)
    super().__setattr__(name, value)

  def __call__(self, *inputs):
    """객체가 호출될 때 함수를 호출하여 forward를 수행하는 메서드

    Args:
      *inputs (Tuple[numpy.adarray | Variable]): 입력 변수들

    Returns:
      Union["Variable", Tuple["Variable"]]: forward 연산 결과
    """
    outputs = self.forward(*inputs) # 순전파 계산
    if not isinstance(outputs, tuple):
      outputs = (outputs,)
    self.inputs = [weakref.ref(x) for x in inputs] # 입력 값 저장
    self.outputs = [weakref.ref(y) for y in outputs] # 출력 값 저장
    return outputs if len(outputs) > 1 else outputs[0]
  
  def forward(self, inputs):
    """forward를 수행하는 메서드

    Args:
      inputs (numpy.ndarray): 입력 데이터

    Raises:
      NotImplementedError: 해당 메서드에서는 기능을 구현하지 않는다.

    Returns:
      numpy.ndarray: forward 연산을 수행한 결과
    """
    raise NotImplementedError()
  
  def params(self):
    """Layer 인스턴스에 포함된 Parameter를 반환하는 메서드"""
    for name in self._params:
      obj = self.__dict__[name]

      if isinstance(obj, Layer):
        yield from obj.params()
      else:
        yield obj
  
  def cleargrads(self):
    """모든 매개변수의 미분값을 재설정하는 메서드"""
    for param in self.params():
      param.cleargrad()

  def save_weights(self, path):
    """가중치를 저장하는 메서드

    Args:
      path (str): 가중치를 저장할 경로
    """
    target_path = os.getcwd() + '/' + 'Aria/assets/cache/' + path
    params_dict = {}
    self._flatten_params(params_dict)
    array_dict = {key: param.data for key, param in params_dict.items() if param is not None}

    try:
      np.savez_compressed(target_path, **array_dict)
    except (Exception, KeyboardInterrupt) as e:
      if os.path.exists(target_path):
        os.remove(target_path)
  
  def load_weights(self, path):
    """가중치를 불러오는 메서드

    Args:
      path (str): 가중치를 불러올 경로
    """
    target_path = os.getcwd() + '/' + 'Aria/assets/cache/' + path
    npz = np.load(target_path)
    params_dict = {}
    self._flatten_params(params_dict)
    for key, param in params_dict.items():
      param.data = npz[key]

  def _flatten_params(self, params_dict, parent_key= ''):
    """Layer에 포함된 모든 Parameter를 한 줄로 평탄화하는 메서드

    Args:
      params_dict (dict): 평탄화된 매개변수를 저장할 딕셔너리
      parent_key (Optional[str]): 부모 키(기본값은 '')
    """
    for name in self._params:
      obj = self.__dict__[name]
      key = parent_key + '/' + name if parent_key else name

      if isinstance(obj, Layer):
        obj._flatten_params(params_dict, key)
      else:
        params_dict[key] = obj