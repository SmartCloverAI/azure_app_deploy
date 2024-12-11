import importlib


MODELS = [
  "anomaly_document_nir"
]




class ModelLoader:  
  def __init__(self, model_name):
    self.model_path = "./models/" + model_name
    self.model_name = model_name
    self.model = None
    self.load_model()
    return
  
  def load_model(self):
    importlib.import_module()
    return