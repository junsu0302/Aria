from Aria.core.Layer import Layer
from Aria.utils.Visualize import plot_dot_graph

class Model(Layer):
  def plot(self, *inputs, to_file='model.png'): # 모델 구조 이미지 반환
    """모델의 구조 이미지를 반환하는 메서드

    Args:
      *inputs (Tuple[numpy.adarray]): 입력 데이터
      to_file (Optional[str]): 모델 구조 이미지를 저장할 파일 경로(기본값은 'model.png')
    """
    y = self.forward(*inputs)
    return plot_dot_graph(y, verbose=True, to_file=to_file)