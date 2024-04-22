import numpy as np

from Aria.core.Model import Model
from Aria.utils.Visualize import plot_dot_graph
from Aria.utils.Download import get_file

import Aria.layers as L
import Aria.activation as AF
import Aria.functions.Convolutional as Conv
import Aria.functions.Tensor as F
import Aria.functions.Train as Train
  
class TwoLayerNet(Model):
  def __init__(self, hidden_size, out_size):
    super().__init__()
    self.l1 = L.Linear(hidden_size)
    self.l2 = L.Linear(out_size)

  def forward(self, x):
    y = AF.sigmoid(self.l1(x))
    y = self.l2(y)
    return y
  
class MLP(Model):
  def __init__(self, fc_output_sizes, activation=AF.sigmoid):
    super().__init__()
    self.activation = activation
    self.layers = []

    for i, out_size in enumerate(fc_output_sizes):
      layer = L.Linear(out_size)
      setattr(self, 'l' + str(i), layer)
      self.layers.append(layer)

  def forward(self, x):
    for l in self.layers[:-1]:
      x = self.activation(l(x))
    return self.layers[-1](x)
  
class VGG16(Model):
  WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz'

  def __init__(self, pretrained=False):
    super().__init__()
    self.conv1_1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
    self.conv1_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
    self.conv2_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.conv2_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.conv3_1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
    self.conv3_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
    self.conv3_3 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.conv4_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.conv4_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
    self.conv4_3 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
    self.conv5_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.conv5_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.conv5_3 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.fc6 = L.Linear(4096)
    self.fc7 = L.Linear(4096)
    self.fc8 = L.Linear(1000)

    if pretrained:
      self.load_weights('vgg16.npz')

  def forward(self, x):
    x = AF.relu(self.conv1_1(x))
    x = AF.relu(self.conv1_2(x))
    x = Conv.pooling(x, 2, 2)
    x = AF.relu(self.conv2_1(x))
    x = AF.relu(self.conv2_2(x))
    x = Conv.pooling(x, 2, 2)
    x = AF.relu(self.conv3_1(x))
    x = AF.relu(self.conv3_2(x))
    x = AF.relu(self.conv3_3(x))
    x = Conv.pooling(x, 2, 2)
    x = AF.relu(self.conv4_1(x))
    x = AF.relu(self.conv4_2(x))
    x = AF.relu(self.conv4_3(x))
    x = Conv.pooling(x, 2, 2)
    x = AF.relu(self.conv5_1(x))
    x = AF.relu(self.conv5_2(x))
    x = AF.relu(self.conv5_3(x))
    x = Conv.pooling(x, 2, 2)
    x = F.reshape(x, (x.shape[0], -1))
    x = Train.dropout(AF.relu(self.fc6(x)))
    x = Train.dropout(AF.relu(self.fc7(x)))
    x = self.fc8(x)
    return x
  
  @staticmethod
  def preprocess(image, size=(224, 224), dtype=np.float32):
    image = image.convert('RGB')
    if size:
      image = image.resize(size)
    image = np.asarray(image, dtype=dtype)
    image = image[:, :, ::-1]
    image -= np.array([103.939, 116.779, 123.68], dtype=dtype)
    image = image.transpose((2, 0, 1))
    return image
  
class SimpleRNN(Model):
  def __init__(self, hidden_size, out_size):
    super().__init__()
    self.rnn = L.RNN(hidden_size)
    self.fc = L.Linear(out_size)

  def reset_state(self):
    self.rnn.reset_state()
    
  def forward(self, x):
    h = self.rnn(x)
    y = self.fc(h)
    return y
  
class SimpleLSTM(Model):
  def __init__(self, hidden_size, out_size):
    super().__init__()
    self.rnn = L.LSTM(hidden_size)
    self.fc = L.Linear(out_size)

  def reset_state(self):
    self.rnn.reset_state()

  def __call__(self, x):
    y = self.rnn(x)
    y = self.fc(y)
    return y