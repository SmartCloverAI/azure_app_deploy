import pickle
import numpy as np
import json
import pandas as pd
import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def preparing_inference_df(request_inference_data, training_columns):
  inference_data_df = pd.DataFrame(request_inference_data,columns = training_columns)

  inference_data_df['SERIE_AVIZ'] = inference_data_df['SERIE_AVIZ'].fillna('NULL')
  inference_data_df['SERIE_AVIZ'] = inference_data_df['SERIE_AVIZ'].astype("string")
  
  OH_df_inference = pd.DataFrame(training_OH_encoder_score.transform(np.array(inference_data_df['SERIE_AVIZ']).reshape(-1,1)),columns = training_OH_encoder_score.get_feature_names_out(['SERIE_AVIZ']))
  OH_df_inference.index = inference_data_df.index

  inference_data_df = pd.concat([inference_data_df, OH_df_inference], axis=1, join="inner")
  inference_data_df = inference_data_df.drop(columns = ['SERIE_AVIZ'])

  inference_data_df = inference_data_df.fillna(0)

  return inference_data_df

def init():
  global autoencoder_loaded_score
  global training_results_score
  global training_scaler_score
  global training_OH_encoder_score

  root = "models"
  model_folder = "anomaly_document_nir"

  model_file = "autoencoder_DOCUMENTE_NIR.keras"


  with open(os.path.join(root,model_folder, "training_results.pkl"), 'rb') as file: 
    training_results_score = pickle.load(file) 

  with open(os.path.join(root,model_folder, "training_scaler.pkl"), 'rb') as file: 
    training_scaler_score = pickle.load(file)

  with open(os.path.join(root, model_folder,"training_OH_encoder.pkl"), 'rb') as file: 
    training_OH_encoder_score = pickle.load(file) 

  autoencoder_loaded_score = tf.keras.models.load_model(os.path.join(root,model_folder, model_file))

def run(raw_data):
  response = {}
  try:
    request_json= json.loads(raw_data)

    inference_data = request_json["data"]
    columns_training_table = training_results_score['columns_training_table']
    
    inference_df = preparing_inference_df(inference_data,columns_training_table)

    inference_data_scaled = training_scaler_score.transform(inference_df)

    inference_reconstruction = autoencoder_loaded_score.predict(inference_data_scaled)
    inference_train_loss_anomaly = tf.keras.losses.mae(inference_reconstruction,inference_data_scaled)

    inference_threshold = training_results_score['threshold']
    inference_results = inference_train_loss_anomaly.numpy() > inference_threshold

    response['response'] = inference_results.tolist()
    response['values'] = inference_train_loss_anomaly.numpy().round(3).tolist()

    return response
      
  except ValueError as e:
    response["response"] = e
    return response
  

if __name__ == "__main__":
  
  init()
  
  # from notebooks
  str_data = '{"data":[[55.0, 2845.0, 163.0, 123.0, null, null, 1.0, 720.0, 36.0], [55.0, 2884.0, 163.0, 123.0, null, null, 1.0, 50.0, 9.5], [55.0, 2857.0, 172.0, 132.0, null, null, 1.0, 10025.0, 1902.25], [55.0, 2886.0, 163.0, 123.0, null, null, 1.0, 250.0, 47.5]]}'
  
  # from site
  str_data = '{"data":[[114,3197,179,140,null,null,1,150,28.5],[114,3166,177,138,null,null,2,150,28.5],[114,3156,177,138,null,null,1,10000,1900]]}'
  
  data = str_data # json.loads(str_data)
    
  endpoint_data_expected = [True, True, True]

  response = run(data)
  
  result = response['response']
  
  print("Expected: ", endpoint_data_expected)
  print("Model:    ", result)

  assert result == endpoint_data_expected
  
  print(response)
  
  