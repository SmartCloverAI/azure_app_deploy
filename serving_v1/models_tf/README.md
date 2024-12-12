# TF Models add procedure


  1. Copy the model folder
  2. Setup the config.json object with the name, folder, etc
  3. Copy from coresponding Azure Notebook and create model.py or whatever.py with 
    - eliminate `azureml` import
    - modify `root` and `model_folder` in `init` method
    - optional add extra output below if tf score


```python
        response['values'] = inference_train_loss_anomaly.numpy().round(3).tolist()
        response['threshold'] = inference_threshold
```