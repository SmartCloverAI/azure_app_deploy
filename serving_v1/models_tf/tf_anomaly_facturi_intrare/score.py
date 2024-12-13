import json
import pickle
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

def preparing_inference_df(request_inference_data, training_columns):
    inference_data_df = pd.DataFrame(request_inference_data,columns = training_columns)

    inference_data_df['DATA_SCADENTA'] = pd.to_datetime(inference_data_df['DATA_SCADENTA'], errors='coerce')
    inference_data_df['year_DATA_SCADENTA'] = inference_data_df['DATA_SCADENTA'].dt.year
    inference_data_df['month_DATA_SCADENTA'] = inference_data_df['DATA_SCADENTA'].dt.month
    inference_data_df['week_DATA_SCADENTA'] = inference_data_df['DATA_SCADENTA'].dt.day_of_week
    inference_data_df['day_DATA_SCADENTA'] = inference_data_df['DATA_SCADENTA'].dt.day

    inference_data_df = inference_data_df.drop(columns = ['DATA_SCADENTA'])
    inference_data_df = inference_data_df.fillna(0)

    return inference_data_df

def init():
    global autoencoder_loaded_score
    global training_results_score
    global training_scaler_score

    root = "models"
    model_folder = "tf_anomaly_facturi_intrare"

    model_file = "autoencoder_DOCUMENTE_FACTURI_INTRARE.keras"



    with open(os.path.join(root,model_folder, "training_results.pkl"), 'rb') as file: 
        training_results_score = pickle.load(file) 

    with open(os.path.join(root,model_folder, "training_scaler.pkl"), 'rb') as file: 
        training_scaler_score = pickle.load(file)

    autoencoder_loaded_score = load_model(os.path.join(root,model_folder, model_file))

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
        response['threshold'] = inference_threshold
        
        return response
        
    except ValueError as e:
        response["response"] = e
        return response