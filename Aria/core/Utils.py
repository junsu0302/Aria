def as_array(x):
  import numpy as np

  if np.isscalar(x):
    return np.array(x)
  return x

def as_varialbe(obj):
  from Aria.core.Variable import Variable
  
  if isinstance(obj, Variable):
    return obj
  return Variable(obj)