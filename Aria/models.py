from Aria.core.Model import Model
from Aria.utils.Visualize import plot_dot_graph

import Aria.layers as L
import Aria.activation as AF
  
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