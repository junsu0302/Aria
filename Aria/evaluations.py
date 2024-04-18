from Aria.core.Variable import Variable
from Aria.core.Utils import as_varialbe, as_array

def accuracy(y, t):
  y, t = as_varialbe(y), as_varialbe(t)

  pred = y.data.argmax(axis=1).reshape(t.shape)
  result = (pred == t.data)
  acc = result.mean()
  return Variable(as_array(acc))