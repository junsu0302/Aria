import os
import subprocess

def _dot_var(v, verbose:bool=False) -> str:
  """Variable 객체를 DOT 형식으로 변환

  Args:
    v (Variable): 변환할 Variable 객체
    verbose (bool, optional): 자세한 정보를 표시할지 여부(기본값은 False)
      
  Returns:
    str: DOT 형식으로 변환된 문자열
  """
  dot_var = '{} [label="{}", color=orange, style=filled]\n'

  name = '' if v.name is None else v.name
  if verbose and v.data is not None:
    if v.name is not None:
      name += ': '
    name += str(v.shape) + ' ' + str(v.dtype)

  return dot_var.format(id(v), name)


def _dot_func(f) -> str:
  """Function 객체를 DOT 형식으로 변환

  Args:
    f (Function): 변환할 Function 객체

  Returns:
    str: DOT 형식으로 변환된 문자열
  """
  dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
  ret = dot_func.format(id(f), f.__class__.__name__)

  dot_edge = '{} -> {}\n'
  for x in f.inputs:
    ret += dot_edge.format(id(x), id(f))
  for y in f.outputs:
    ret += dot_edge.format(id(f), id(y()))
  return ret


def get_dot_graph(output, verbose:bool=True) -> str:
  """계산 그래프를 DOT 형식으로 변환

  Args:
    output (Variable): 출력 변수
    verbose (bool, optional): 자세한 정보를 표시할지 여부(기본값은 True)

  Returns:
    str: DOT 형식으로 변환된 계산 그래프
  """
  txt = ''
  funcs = []
  seen_set = set()

  def add_func(f):
    if f not in seen_set:
      funcs.append(f)
      # funcs.sort(key=lambda x: x.generation)
      seen_set.add(f)

  add_func(output.creator)
  txt += _dot_var(output, verbose)

  while funcs:
    func = funcs.pop()
    txt += _dot_func(func)
    for x in func.inputs:
      txt += _dot_var(x, verbose)

      if x.creator is not None:
        add_func(x.creator)

  return 'digraph g {\n' + txt + '}'


def plot_dot_graph(output, verbose:bool=True, to_file:str='graph.png') -> None:
  """계산 그래프를 이미지로 표시하고 DOT 파일로 저장

  Args:
    output (Variable): 출력 변수
    verbose (bool, optional): 자세한 정보를 표시할지 여부(기본값은 True)
    to_file (str, optional): 저장할 파일 이름(기본값은 'graph.png')
  """
  dot_graph = get_dot_graph(output, verbose)

  target_dir = os.getcwd() + '/' + 'Aria/assets/models'
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)
  graph_path = os.path.join(target_dir, 'tmp_graph.dot')
  file_path = os.path.join(target_dir, to_file)

  with open(graph_path, 'w') as f:
    f.write(dot_graph)

  extension = os.path.splitext(to_file)[1][1:]
  cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, file_path)
  subprocess.run(cmd, shell=True)
  os.remove(graph_path) # DOT 파일은 제거
  