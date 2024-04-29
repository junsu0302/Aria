import os
import urllib.request

def show_progress(block_num:int, block_size:int, total_size:int) -> None:
  """다운로드 진행 사항 표시

  Args:
    block_num (int): 현재 받은 블록의 수
    block_size (int): 각 블록의 크기
    total_size (int): 전체 파일의 크기
  """
  bar_template = "\r[{}] {:.2f}%"

  downloaded = block_num * block_size
  p = downloaded / total_size * 100
  i = int(downloaded / total_size * 30)
  if p >= 100.0: p = 100.0
  if i >= 30: i = 30
  bar = "#" * i + "." * (30 - i)
  print(bar_template.format(bar, p), end='')


cache_dir = os.getcwd() + '/' + 'Aria/assets/cache/' 

def get_file(url:str, file_name:str=None) -> str:
  """주어진 URL에서 파일을 다운로드하고 파일 경로 반환

  Args:
    url (str): 다운로드할 파일의 URL
    file_name (str, optional): 저장할 파일의 이름(기본값은 None)

  Returns:
    str: 다운로드한 파일의 경로
  """
  if file_name is None:
    file_name = url[url.rfind('/') + 1:]
  file_path = os.path.join(cache_dir, file_name)

  if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

  if os.path.exists(file_path):
    return file_path

  print("Downloading: " + file_name)
  try:
    urllib.request.urlretrieve(url, file_path, show_progress)
  except (Exception, KeyboardInterrupt) as e:
    if os.path.exists(file_path):
      os.remove(file_path)
    raise
  print(" Done")

  return file_path