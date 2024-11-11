import os
import time


from utils import log_with_color

DEMO_TIME = 20
DEMO_CACHE = "./_cache"
DEMO_CACHE_FN = "test.txt"



def main():
  if not os.path.exists(DEMO_CACHE):
    raise Exception(f"Cache directory {DEMO_CACHE} not found")
  fn = os.path.join(DEMO_CACHE, DEMO_CACHE_FN)
  log_with_color(f"Cache file: {fn}", color="gray")
  try:
    with open(fn, "rt") as f:
      content = f.read()
      log_with_color(f"Cache file content:\n{content}", color="g")
    with open(fn, "a") as f:
      f.write("Demo run.")
  except Exception as e:
    log_with_color(f"Error reading/writing cache file: {e}", color='r')
  
  for i in range(DEMO_TIME):
    if i % 5 == 0:
      log_with_color(f"Processing step {i}", color="light")
    time.sleep(1)


if __name__ == "__main__":
  main()