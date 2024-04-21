import os
import urllib.request

def show_progress(block_num, block_size, total_size):
  bar_template = "\r[{}] {:.2f}%"

  downloaded = block_num * block_size
  p = downloaded / total_size * 100
  i = int(downloaded / total_size * 30)
  if p >= 100.0: p = 100.0
  if i >= 30: i = 30
  bar = "#" * i + "." * (30 - i)
  print(bar_template.format(bar, p), end='')


cache_dir = os.getcwd() + '/' + 'Aria/assets/cache/' 

def get_file(url, file_name=None):
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