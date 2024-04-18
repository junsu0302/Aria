from Aria.core.Layer import Layer
from Aria.utils.Visualize import plot_dot_graph

class Model(Layer):
  def plot(self, *inputs, to_file='model.png'): # 모델 구조 이미지 반환
    y = self.forward(*inputs)
    return plot_dot_graph(y, verbose=True, to_file=to_file)