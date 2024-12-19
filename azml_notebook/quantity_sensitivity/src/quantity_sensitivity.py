import pandas as pd
import numpy as np
import os
import argparse
import pickle
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

print("TensorFlow version:", tf.__version__)

mlflow.start_run()
mlflow.tensorflow.autolog()

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_data_train",
    type=str,
    dest="input_data_train",
    help="Input data train"
)
parser.add_argument(
    "--input_data_test",
    type=str,
    dest="input_data_test",
    help="Input data test"
)

parser.add_argument(
    "--epochs",
    type=int,
    dest="epochs",
    default=100,
    help="Number of epochs",
)
parser.add_argument(
    "--batch_size",
    type=int,
    dest="batch_size",
    default=32,
    help="Number of batch-sizes",
)
parser.add_argument(
    "--first_layer_neurons",
    type=int,
    dest="n_hidden_1",
    default=128,
    help="# of neurons in the first layer",
)
parser.add_argument(
    "--second_layer_neurons",
    type=int,
    dest="n_hidden_2",
    default=64,
    help="# of neurons in the second layer",
)
parser.add_argument(
    "--third_layer_neurons",
    type=int,
    dest="n_hidden_3",
    default=64,
    help="# of neurons in the third layer",
)

parser.add_argument(
    "--registered_model_name", 
    type=str, 
    dest = "registered_model_name",
    help="Name of the model."
    )

args = parser.parse_args()

input_data_train = args.input_data_train
print("Data folder train:", input_data_train)

input_data_test = args.input_data_test
print("Data folder test:", input_data_test)

print(f"Batch size: {args.batch_size}")
print(f"Epochs: {args.epochs}")

print(f"First layer number of neurons: {args.n_hidden_1}")
print(f"Second layer number of neurons: {args.n_hidden_2}")
print(f"Third layer number of neurons: {args.n_hidden_3}")

print("Name of the model:",args.registered_model_name)

mlflow.set_experiment(args.registered_model_name)

#Pregatirea setului de antrenament
quantity_sensitivity_train_df = pd.read_csv(args.input_data_train)

training_results = {}
columns_training_table = list(quantity_sensitivity_train_df.columns)
columns_training_table = [x for x in columns_training_table if x not in ['VENIT','CHELTUIALA','TVA IN','TVA OUT','CASHFLOW']]

training_results['columns_training_table'] = columns_training_table

quantity_sensitivity_train_df['START_DATE_WEEK'] = pd.to_datetime(quantity_sensitivity_train_df['START_DATE_WEEK'], errors='coerce')
quantity_sensitivity_train_df['year'] = quantity_sensitivity_train_df['START_DATE_WEEK'].dt.year
quantity_sensitivity_train_df['month'] = quantity_sensitivity_train_df['START_DATE_WEEK'].dt.month
quantity_sensitivity_train_df['week'] = quantity_sensitivity_train_df['START_DATE_WEEK'].dt.day_of_week
quantity_sensitivity_train_df['day'] = quantity_sensitivity_train_df['START_DATE_WEEK'].dt.day

quantity_sensitivity_train_df = quantity_sensitivity_train_df.drop(columns = ['START_DATE_WEEK'])
quantity_sensitivity_train_df = quantity_sensitivity_train_df.fillna(0)

#Pregatirea setului de test
quantity_sensitivity_test_df = pd.read_csv(args.input_data_test)

quantity_sensitivity_test_df['START_DATE_WEEK'] = pd.to_datetime(quantity_sensitivity_test_df['START_DATE_WEEK'], errors='coerce')
quantity_sensitivity_test_df['year'] = quantity_sensitivity_test_df['START_DATE_WEEK'].dt.year
quantity_sensitivity_test_df['month'] = quantity_sensitivity_test_df['START_DATE_WEEK'].dt.month
quantity_sensitivity_test_df['week'] = quantity_sensitivity_test_df['START_DATE_WEEK'].dt.day_of_week
quantity_sensitivity_test_df['day'] = quantity_sensitivity_test_df['START_DATE_WEEK'].dt.day

quantity_sensitivity_test_df = quantity_sensitivity_test_df.drop(columns = ['START_DATE_WEEK'])
quantity_sensitivity_test_df = quantity_sensitivity_test_df.fillna(0)

y_train = quantity_sensitivity_train_df[['VENIT','CHELTUIALA','TVA IN','TVA OUT','CASHFLOW']]
X_train = quantity_sensitivity_train_df.drop(columns = ['VENIT','CHELTUIALA','TVA IN','TVA OUT','CASHFLOW'])

y_test = quantity_sensitivity_test_df[['VENIT','CHELTUIALA','TVA IN','TVA OUT','CASHFLOW']]
X_test = quantity_sensitivity_test_df.drop(columns = ['VENIT','CHELTUIALA','TVA IN','TVA OUT','CASHFLOW'])

scaler = MinMaxScaler()
training_scaler = scaler.fit(X_train) 
X_train_scaled = training_scaler.transform(X_train)
X_test_scaled = training_scaler.transform(X_test)

price_sensitivity_model = tf.keras.models.Sequential()
price_sensitivity_model.add(tf.keras.layers.Dense(args.n_hidden_1,input_dim = X_train_scaled.shape[1],activation='elu'))
price_sensitivity_model.add(tf.keras.layers.Dropout(.2))
price_sensitivity_model.add(tf.keras.layers.Dense(args.n_hidden_2,activation='elu'))
price_sensitivity_model.add(tf.keras.layers.Dropout(.2))
price_sensitivity_model.add(tf.keras.layers.Dense(args.n_hidden_3,activation='elu'))
price_sensitivity_model.add(tf.keras.layers.Dropout(.2))
price_sensitivity_model.add(tf.keras.layers.Dense(y_train.shape[1],kernel_initializer = 'normal',activation='linear'))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
price_sensitivity_model.compile(optimizer='adam', loss="mae")
history = price_sensitivity_model.fit(X_train_scaled, y_train, epochs=args.epochs, batch_size=args.batch_size,
                    callbacks=[early_stopping],validation_data=(X_test_scaled, y_test)
                    )

print(history.history)

#Saving the results & model

output_model_path = "./outputs/"
os.makedirs(output_model_path, exist_ok=True)


training_scaler_path = output_model_path + '/training_scaler.pkl'
training_results_path = output_model_path + '/training_results.pkl'


with open(training_scaler_path, 'wb') as file: 
    pickle.dump(training_scaler, file)

with open(training_results_path, 'wb') as file: 
    pickle.dump(training_results, file)


price_sensitivity_model.save(output_model_path + "quantity_sensitivity.keras")

mlflow.end_run()