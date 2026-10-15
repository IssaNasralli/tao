import tensorflow as tf
import numpy as np
import os
import random
import itertools
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K


import csv_features 
import training_history as th
import ato 
def get_trainable_params(model):
    return int(np.sum([K.count_params(w) for w in model.trainable_weights]))


if __name__ == "__main__":

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    sequence_length=45
    dataset="jaad"
    feature_path = f'data{sequence_length}_{dataset}/features.csv'
    angle_path = f'data{sequence_length}_{dataset}/angle.csv'
    pose_path = f'data{sequence_length}_{dataset}/pose.csv'
    x_traject_path = f'data{sequence_length}_{dataset}/x_traject.csv'
    y_traject_path = f'data{sequence_length}_{dataset}/y_traject.csv'
    training_results_file = f'training_history_{sequence_length}/training_results_{dataset}.txt'
    # Load and preprocess the data
    feature_max=6.0
    angle_max=180
    frame_width, frame_height=1920.0, 1080.0

    features, poses, angles, bbxes, bbxes_future, action_labels = csv_features.load_and_preprocess_data(feature_path,
    angle_path, pose_path, x_traject_path, y_traject_path,feature_max, angle_max, frame_width, frame_height,sequence_length=sequence_length)
    
 
    # Define the ranges for the parameters
    unit_lstm_values = [1024,512,256,128,64] 
    batch_size_values = [512, 256,192, 160, 96, 128, 64]
    loss_weight = [1000,900,800,700,600,500,400,300,200,100,1] 

    epochs=1000

    # Open a text file to save training results
    iterated_values = th.read_training_results(training_results_file)
    print("iterated:", iterated_values)
    # Assuming the file exist with head : LSTM_units, batch_size, weight, loss, loss_bbox_output, loss_pedestrian_classification
    with open(training_results_file, 'a') as f:
        
        # Iterate through all combinations of the parameter values
        for unit_lstm, batch_size, loss_weight in itertools.product(unit_lstm_values, batch_size_values, loss_weight):
            
            if ((unit_lstm, batch_size, loss_weight) in iterated_values)==True:
                print(f"Skiping with LSTM_units={unit_lstm}, batch_size={batch_size}, weight={loss_weight}")
                continue
            else:
                print(f"These combination values do not exist: LSTM_units={unit_lstm}, batch_size={batch_size}, weight={loss_weight}")


            print(f"Training with LSTM_units={unit_lstm}, batch_size={batch_size}, weight={loss_weight}")
            
            # Create or load the model
            model = ato.build_model(unit_lstm, loss_weight)
            print("Nb parameter:", get_trainable_params(model))

            model_json = model.to_json()
            with open('model_architecture.json', 'w') as json_file:
                json_file.write(model_json)                
            # Train the model and get the history
            history = ato.train_model(dataset, sequence_length, model, features, poses, angles, bbxes, bbxes_future, action_labels,unit_lstm, batch_size,loss_weight, epochs, seed)
            
            # Extract final loss values
            final_loss = history.history['loss'][-1] if 'loss' in history.history else np.nan
            final_loss_bbox_output = history.history['bbox_output_loss'][-1] if 'bbox_output_loss' in history.history else np.nan
            final_loss_pedestrian_classification = history.history['pedestrian_classification_loss'][-1] if 'pedestrian_classification_loss' in history.history else np.nan
            
            print(f"Final loss: {final_loss}, Final bbox_output loss: {final_loss_bbox_output}, Final pedestrian_classification loss: {final_loss_pedestrian_classification}")
            
            # Save the results to the text file
            f.write(f"{unit_lstm}, {batch_size}, {loss_weight}, {final_loss}, {final_loss_bbox_output}, {final_loss_pedestrian_classification}\n")
            f.flush()  # Ensure the data is written to the file
            os.fsync(f.fileno())
            # Plot and save the training history
            th.plot_and_save_history(history, unit_lstm, batch_size, loss_weight)
            

    
    print("Training complete and results saved to text file.")
