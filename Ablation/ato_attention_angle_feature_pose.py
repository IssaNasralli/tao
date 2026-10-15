from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Attention, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
import os
import numpy as np
import csv
from tensorflow.keras.layers import TimeDistributed
import tensorflow as tf


def create_inputs():
    """
    Create input layers for features, pose estimation, angles, and bounding boxes.
    """
    bbox_input = Input(shape=(None, 4), name='bbox_input')
    return bbox_input

def bbox_processing(bbox_input, unit_lstm, output_steps=45):
    """
    Encode and decode bounding boxes to predict future bounding boxes, with a longer sequence.
    """
    # Encode the bounding box input sequence (e.g., input 15 bounding boxes)
    bbox_encoder = LSTM(unit_lstm, return_state=True, return_sequences=True, name='bbox_encoder')
    bbox_encoded_output, state_h, state_c = bbox_encoder(bbox_input)

        
    
    decoder_input = tf.zeros((tf.shape(bbox_input)[0], output_steps, tf.shape(bbox_input)[-1]))  # Shape: (batch_size, 45, 4)
    
    # Decode bounding boxes (set return_sequences=True for 45 steps)
    bbox_decoder = LSTM(unit_lstm, return_sequences=True, name='bbox_decoder')

    # Predict 45 bounding boxes using the  decoder input
    future_bbox_prediction = bbox_decoder(decoder_input)
    
    # Apply a time-distributed dense layer to generate the final bounding box coordinates for 45 time steps
    future_bbox_prediction = TimeDistributed(Dense(4, activation='linear'), name='bbox_output')(future_bbox_prediction)
    
    return future_bbox_prediction

def build_model(unit_lstm,loss_weight):
    """
    Build the model with all components.
    """
    bbox_input = create_inputs()
    print("bbox_input.shape: ",bbox_input.shape)
    
    
    future_bbox_prediction = bbox_processing(bbox_input, unit_lstm, output_steps=45)

    print("future_bbox_prediction:", future_bbox_prediction.shape)


    model = Model(inputs=[bbox_input],
                  outputs=[future_bbox_prediction])
    model.compile(optimizer=Adam(),
                  loss={'bbox_output': 'mse'},
                  loss_weights={'bbox_output': 1.0}
                )

    return model


def train_model(sequence_length, model, X_feature, X_pose, X_angle, X_bbox, Y_bbox, Y_p_s, unit_lstm, batch_size, loss_weight, epochs, random_state):
    # Define the file name based on the given parameters
    weight_filename = f"{sequence_length}frames_unit_{unit_lstm}_number1.h5"
    
    # Check if the best weight file exists
    if os.path.exists(weight_filename):
        print(f"Loading weights from {weight_filename}")
        model.load_weights(weight_filename)
    
    # First split: Train+Val (60%) and Test (40%)
    X_feature_train_val, X_feature_test, X_pose_train_val, X_pose_test, X_angle_train_val, X_angle_test, X_bbox_train_val, X_bbox_test, Y_bbox_train_val, Y_bbox_test, Y_p_s_train_val, Y_p_s_test = train_test_split(
        X_feature, X_pose, X_angle, X_bbox, Y_bbox, Y_p_s, test_size=0.4, random_state=random_state)
    
    # Second split: Train (50%) and Val (10%) from Train+Val (60%)
    X_feature_train, X_feature_val, X_pose_train, X_pose_val, X_angle_train, X_angle_val, X_bbox_train, X_bbox_val, Y_bbox_train, Y_bbox_val, Y_p_s_train, Y_p_s_val = train_test_split(
        X_feature_train_val, X_pose_train_val, X_angle_train_val, X_bbox_train_val, Y_bbox_train_val, Y_p_s_train_val, test_size=1/6, random_state=random_state)  # 1/6 * 60% ≈ 10%

    # Define the callbacks for early stopping and model checkpointing
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(weight_filename, monitor='val_loss', save_best_only=True, mode='min')

    history = model.fit(
        [X_bbox_train],
        [Y_bbox_train],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([X_bbox_val], [Y_bbox_val]),
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Save the final model weights with the dynamic filename
    final_weight_filename = f"{sequence_length}frames_unit_{unit_lstm}_batch_{batch_size}_loss_{loss_weight}_number1.h5"
    model.save_weights(final_weight_filename)

    # Evaluate the model on the test set
    test_loss = model.evaluate([X_bbox_test], [Y_bbox_test])
    print(f"Test loss: {test_loss}")
    
    return history

def calculate_bbox_centers(predictions):
    bbox_centers = []
    for i in range(len(predictions)):
        for bbox in predictions[i]:
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            bbox_centers.append([x_center, y_center])
    return bbox_centers


def evaluate_model_per_sample_center(seq_lenght, model, X_feature, X_pose, X_angle, X_bbox, Y_bbox, Y_p_s, unit_lstm, batch_size, loss_weight, epochs, random_state):
    weight_filename = f"{seq_lenght}frames_unit_{unit_lstm}_batch_{batch_size}_loss_{loss_weight}_number1.h5"
    
    if os.path.exists(weight_filename):
        print(f"Loading weights from {weight_filename}")
        model.load_weights(weight_filename)
    else:
        print(f"Weight file {weight_filename} not found.")
        return

    total_samples = len(X_feature)
    
    csv_filename = f"bbox_loss_per_sample_{seq_lenght}_unit_{unit_lstm}_batch_{batch_size}_loss_weight_{loss_weight}_number1.csv"
    
    if os.path.exists(csv_filename):
        with open(csv_filename, mode='r') as file:
            reader = csv.reader(file)
            lines = list(reader)
            l_saved = len(lines) - 1
        print(f"Resuming from sample {l_saved + 1}")
    else:
        l_saved = 0
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Sample Index', 'Bbx Loss 0.5s','Bbx Loss 1s','Bbx Loss 1.5s','Bbx Loss 1.5s (evaluate)', 'C_MSE', 'CF_MSE'])

    total_bbox_loss_1_5_sec = total_bbox_loss_1_sec = total_bbox_loss_0_5_sec = total_C_MSE = total_CF_MSE = total_pedestrian_class_loss =total_overall_loss= 0
    
    for i in range(l_saved, total_samples):
        X_feature_sample = np.expand_dims(X_feature[i], axis=0)
        X_pose_sample = np.expand_dims(X_pose[i], axis=0)
        X_angle_sample = np.expand_dims(X_angle[i], axis=0)
        X_bbox_sample = np.expand_dims(X_bbox[i], axis=0)
        Y_bbox_sample = np.expand_dims(Y_bbox[i], axis=0)
        Y_p_s_sample = np.expand_dims(Y_p_s[i], axis=0)

        ground_truth_centers = calculate_bbox_centers(Y_bbox_sample)
        
        prediction = model.predict([X_bbox_sample], verbose=0)
        predicted_centers = calculate_bbox_centers(prediction[0])

        bbox_loss_1_5_sec_evaluate = model.evaluate(
            [X_bbox_sample],
            [Y_bbox_sample],
            verbose=0
        )
        if np.isnan(overall_loss):
            total_samples=total_samples-1
            continue
        
        predicted_bbox = prediction[0] 
        
        
        
        # Bounding Box Loss at 0.5s (First 15 frames)
        bbox_loss_0_5_sec = np.mean(np.square(Y_bbox_sample[:, :15, :] - predicted_bbox[:, :15, :]))
        
        # Bounding Box Loss at 1s (First 30 frames)
        bbox_loss_1_sec = np.mean(np.square(Y_bbox_sample[:, :30, :] - predicted_bbox[:, :30, :]))
        
        # Bounding Box Loss at 1.5s (Existing code, full sequence of 45 frames)
        bbox_loss_1_5_sec = np.mean(np.square(Y_bbox_sample[:, :45, :] - predicted_bbox[:, :45, :]))

            
        # **C_MSE Calculation**
        C_MSE = np.mean((np.array(predicted_centers) - np.array(ground_truth_centers)) ** 2)

        # **CF_MSE Calculation** (Final frame MSE)
        CF_MSE = np.mean((np.array(predicted_centers[-1]) - np.array(ground_truth_centers[-1])) ** 2)

        total_C_MSE += C_MSE
        total_CF_MSE += CF_MSE
        total_bbox_loss_1_5_sec += bbox_loss_1_5_sec
        total_bbox_loss_1_sec += bbox_loss_1_sec
        total_bbox_loss_0_5_sec += bbox_loss_0_5_sec

        # Append current metrics to CSV
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i + 1, bbox_loss_0_5_sec, bbox_loss_1_sec, bbox_loss_1_5_sec, bbox_loss_1_5_sec_evaluate, C_MSE, CF_MSE])
            
        print(f"{i+1}/{total_samples}")

    average_bbox_loss_1_5_sec = total_bbox_loss_1_5_sec / total_samples
    average_bbox_loss_1_sec = total_bbox_loss_1_sec / total_samples
    average_bbox_loss_0_5_sec = total_bbox_loss_0_5_sec / total_samples
    
    average_C_MSE = total_C_MSE /  total_samples
    average_CF_MSE = total_CF_MSE / total_samples
    
    print(f"average_bbox_loss_1_5_sec: {average_bbox_loss_1_5_sec}")
    print(f"average_bbox_loss_1_sec: {average_bbox_loss_1_sec}")
    print(f"average_bbox_loss_0_5_sec: {average_bbox_loss_0_5_sec}")
    print(f"Average C_MSE: {average_C_MSE}")
    print(f"Average CF_MSE: {average_CF_MSE}")
    
    return average_bbox_loss_1_5_sec, average_bbox_loss_1_sec, average_bbox_loss_0_5_sec, average_C_MSE, average_CF_MSE




