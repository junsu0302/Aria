import numpy as np
from collections import OrderedDict

from proto.maths import numerical_gradient
from proto.layers import Convolution, Pooling, Linear, SoftmaxWithLoss
from proto.activations import Relu

class SimpleConvNet:
  def __init__(self, input_dim=(1, 28, 28), 
                conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                hidden_size=100, output_size=10, weight_init_std=0.01):
    filter_num = conv_param['filter_num']
    filter_size = conv_param['filter_size']
    filter_pad = conv_param['pad']
    filter_stride = conv_param['stride']
    input_size = input_dim[1]
    conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
    pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

    # 가중치 초기화
    self.params = {}
    self.params['W1'] = weight_init_std * \
                          np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
    self.params['b1'] = np.zeros(filter_num)
    self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
    self.params['b2'] = np.zeros(hidden_size)
    self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
    self.params['b3'] = np.zeros(output_size)

    # 계층 생성
    self.layers = OrderedDict()
    self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
    self.layers['Relu1'] = Relu()
    self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
    self.layers['Linear1'] = Linear(self.params['W2'], self.params['b2'])
    self.layers['Relu2'] = Relu()
    self.layers['Linear2'] = Linear(self.params['W3'], self.params['b3'])

    self.last_layer = SoftmaxWithLoss()

  def predict(self, x):
    for layer in self.layers.values():
      x = layer.forward(x)

    return x

  def loss(self, x, t):
    y = self.predict(x)
    return self.last_layer.forward(y, t)

  def accuracy(self, x, t, batch_size=100):
    if t.ndim != 1 : t = np.argmax(t, axis=1)
        
    acc = 0.0
        
    for i in range(int(x.shape[0] / batch_size)):
      tx = x[i*batch_size:(i+1)*batch_size]
      tt = t[i*batch_size:(i+1)*batch_size]
      y = self.predict(tx)
      y = np.argmax(y, axis=1)
      acc += np.sum(y == tt) 
        
    return acc / x.shape[0]

  def numerical_gradient(self, x, t):
    loss_w = lambda w: self.loss(x, t)

    grads = {}
    for idx in (1, 2, 3):
      grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
      grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

    return grads

  def gradient(self, x, t):
    # forward
    self.loss(x, t)

    # backward
    dout = 1
    dout = self.last_layer.backward(dout)

    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
      dout = layer.backward(dout)

    # 결과 저장
    grads = {}
    grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
    grads['W2'], grads['b2'] = self.layers['Linear1'].dW, self.layers['Linear1'].db
    grads['W3'], grads['b3'] = self.layers['Linear2'].dW, self.layers['Linear2'].db

    return grads