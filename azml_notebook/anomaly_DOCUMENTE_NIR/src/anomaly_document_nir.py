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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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
    help="Batch size",
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
print("Name of the model:",args.registered_model_name)

mlflow.set_experiment(args.registered_model_name)

#Pregatirea setului de antrenament
anomaly_train_df = pd.read_csv(args.input_data_train,index_col=0)

training_results = {}
columns_training_table = list(anomaly_train_df.columns)
training_results['columns_training_table'] = columns_training_table

anomaly_train_df['SERIE_AVIZ'] = anomaly_train_df['SERIE_AVIZ'].fillna('NULL')
anomaly_train_df['SERIE_AVIZ'] = anomaly_train_df['SERIE_AVIZ'].astype("string")

OH_encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
OH_df_train = pd.DataFrame(OH_encoder.fit_transform(np.array(anomaly_train_df['SERIE_AVIZ']).reshape(-1,1)),columns = OH_encoder.get_feature_names_out(['SERIE_AVIZ']))
OH_df_train.index = anomaly_train_df.index

anomaly_train_df = pd.concat([anomaly_train_df, OH_df_train], axis=1, join="inner")
anomaly_train_df = anomaly_train_df.drop(columns = ['SERIE_AVIZ'])

anomaly_train_df = anomaly_train_df.fillna(0)

#Pregatirea setului de test
anomaly_test_df = pd.read_csv(input_data_test,index_col=0)

anomaly_test_df['SERIE_AVIZ'] = anomaly_test_df['SERIE_AVIZ'].fillna('NULL')
anomaly_test_df['SERIE_AVIZ'] = anomaly_test_df['SERIE_AVIZ'].astype("string")

OH_df_test = pd.DataFrame(OH_encoder.transform(np.array(anomaly_test_df['SERIE_AVIZ']).reshape(-1,1)),columns = OH_encoder.get_feature_names_out(['SERIE_AVIZ']))
OH_df_test.index = anomaly_test_df.index

anomaly_test_df = pd.concat([anomaly_test_df, OH_df_test], axis=1, join="inner")
anomaly_test_df = anomaly_test_df.drop(columns = ['SERIE_AVIZ'])

anomaly_test_df = anomaly_test_df.fillna(0)

scaler = MinMaxScaler()
training_scaler = scaler.fit(anomaly_train_df) 
X_train_scaled = training_scaler.transform(anomaly_train_df)
X_test_scaled = training_scaler.transform(anomaly_test_df)

input_dim = X_train_scaled.shape[1]
autoencoder = tf.keras.models.Sequential([
    
    # deconstruct / encode
    tf.keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim, )), 
    tf.keras.layers.Dense(32, activation='elu'),
    tf.keras.layers.Dense(16, activation='elu'),
    tf.keras.layers.Dense(8, activation='elu'),
    tf.keras.layers.Dense(4, activation='elu'),
    
    # reconstruction / decode
    tf.keras.layers.Dense(8, activation='elu'),
    tf.keras.layers.Dense(16, activation='elu'),
    tf.keras.layers.Dense(32, activation='elu'),
    tf.keras.layers.Dense(input_dim, activation='elu')
    
])

autoencoder.compile(optimizer="adam", 
                    loss="mae",
                    metrics=["acc"])

autoencoder.compile(optimizer="adam", 
                    loss="mae",
                    metrics=["acc"])

print(autoencoder.summary())

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=5,
    verbose=1, 
    mode='min',
    restore_best_weights=True
)

history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    shuffle=True,
    epochs=args.epochs,
    batch_size=args.batch_size,
    callbacks = [early_stop]
)

reconstruction = autoencoder.predict(X_test_scaled)
train_loss = tf.keras.losses.mae(reconstruction, X_test_scaled)
threshold = np.mean(train_loss) + 2*np.std(train_loss)

training_results['threshold'] = threshold
training_results['train_loss'] = train_loss

#Saving the results & model

output_model_path = "./outputs/"
os.makedirs(output_model_path, exist_ok=True)


training_scaler_path = output_model_path + '/training_scaler.pkl'
training_results_path = output_model_path + '/training_results.pkl'
training_OH_encoder_path = output_model_path + '/training_OH_encoder.pkl'


with open(training_scaler_path, 'wb') as file: 
    pickle.dump(training_scaler, file)

with open(training_results_path, 'wb') as file: 
    pickle.dump(training_results, file)

with open(training_OH_encoder_path, 'wb') as file: 
    pickle.dump(OH_encoder, file)

autoencoder.save(output_model_path + "autoencoder_DOCUMENTE_NIR.keras")

mlflow.end_run()