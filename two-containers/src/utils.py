from datetime import datetime

from config import APP_NAME

def log_with_color(message, color="d", test=False):
  """
    Log message with color in the terminal.
    :param message: Message to log
    :param color: Color of the message
  """
  color_codes = {
    'n': "\x1b[1;37m", # normal white
    "d": "\033[90m", # dark gray

    "y": "\033[93m",
    "r": "\033[91m",
    
    "g": "\033[92m",
    
    'b': "\x1b[1;36m",    
    'm': "\x1b[1;35m",    
    }
  end_color = "\033[0m"
  color = color.lower()[:1]
  date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  message = f"[{APP_NAME}][{date}] {message}"
  if test:
    for color in color_codes:
      print(f"{color_codes.get(color, '')}{message}{end_color} {color}", flush=True)
  else:
    print(f"{color_codes.get(color, '')}{message}{end_color}", flush=True)  
  return