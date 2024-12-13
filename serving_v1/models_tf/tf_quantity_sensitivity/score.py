import json
import pickle
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model


def preparing_inference_df(request_inference_data, training_columns):
    inference_data_df = pd.DataFrame(request_inference_data,columns = training_columns)

    inference_data_df['START_DATE_WEEK'] = pd.to_datetime(inference_data_df['START_DATE_WEEK'], errors='coerce')
    inference_data_df['year'] = inference_data_df['START_DATE_WEEK'].dt.year
    inference_data_df['month'] = inference_data_df['START_DATE_WEEK'].dt.month
    inference_data_df['week'] = inference_data_df['START_DATE_WEEK'].dt.day_of_week
    inference_data_df['day'] = inference_data_df['START_DATE_WEEK'].dt.day

    inference_data_df = inference_data_df.drop(columns = ['START_DATE_WEEK'])
    inference_data_df = inference_data_df.fillna(0)

    return inference_data_df

def init():
    global quantity_sensitivity_model_score
    global training_results_score
    global training_scaler_score

    root = "models"
    model_folder = "tf_quantity_sensitivity"

    model_file = "quantity_sensitivity.keras"



    with open(os.path.join(root,model_folder, "training_results.pkl"), 'rb') as file: 
        training_results_score = pickle.load(file) 

    with open(os.path.join(root,model_folder, "training_scaler.pkl"), 'rb') as file: 
        training_scaler_score = pickle.load(file)

    quantity_sensitivity_model_score = load_model(os.path.join(root,model_folder, model_file))

def run(raw_data):
    response = {}
    try:
        request_json= json.loads(raw_data)

        inference_data = request_json["data"]
        columns_training_table = training_results_score['columns_training_table']
        
        inference_df = preparing_inference_df(inference_data,columns_training_table)

        inference_data_scaled = training_scaler_score.transform(inference_df)

        inference_results = quantity_sensitivity_model_score.predict(inference_data_scaled)

        response['response'] = inference_results.tolist()

        return response
        
    except ValueError as e:
        response["response"] = e
        return response