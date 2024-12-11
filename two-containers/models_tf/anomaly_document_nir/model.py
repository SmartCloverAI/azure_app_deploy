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

def init(**kwargs):
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
  
  return

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
  