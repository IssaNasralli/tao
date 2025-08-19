from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Attention, MultiHeadAttention, Multiply, Add, Reshape
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
    feature_input = Input(shape=(None, 32), name='feature_input')
    pose_input = Input(shape=(None, 24), name='pose_input')
    angle_input = Input(shape=(None, 1), name='angle_input')
    bbox_input = Input(shape=(None, 4), name='bbox_input')
    return feature_input, pose_input, angle_input, bbox_input


def feature_encoding(feature_input, unit_lstm, dropout_rate=0.5, l2_reg=0.01):
    """
    Encode the features with self-attention and LSTM with dropout and L2 regularization.
    """
    # Self-Attention
    feature_self_attention = MultiHeadAttention(num_heads=2, key_dim=16, name='feature_self_attention')(feature_input, feature_input)
    
    # Apply Batch Normalization
    feature_self_attention = BatchNormalization()(feature_self_attention)
    
    # LSTM with L2 Regularization and Dropout
    lstm_layer = LSTM(unit_lstm, return_sequences=False, return_state=True, 
                      kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg), 
                      dropout=dropout_rate, recurrent_dropout=dropout_rate, 
                      name='feature_encoder')
    feature_encoder, state_h, state_c = lstm_layer(feature_self_attention)
    
    return feature_encoder, state_h, state_c

def pose_encoding(pose_input, feature_hidden_state, feature_cell_state, unit_lstm, dropout_rate=0.5, l2_reg=0.01):
    """
    Encode the pose with self-attention and LSTM with dropout and L2 regularization.
    """
    # Initial hidden and cell state
    initial_state = [feature_hidden_state, feature_hidden_state]
    
    # Self-Attention
    pose_self_attention = MultiHeadAttention(num_heads=2, key_dim=16, name='pose_self_attention')(pose_input, pose_input)
    
    # Apply Batch Normalization
    pose_self_attention = BatchNormalization()(pose_self_attention)
    
    # LSTM with L2 Regularization and Dropout
    lstm_layer = LSTM(unit_lstm, return_sequences=False, return_state=True, 
                      kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg), 
                      dropout=dropout_rate, recurrent_dropout=dropout_rate, 
                      name='pose_encoder')
    pose_encoder, pose_state_h, pose_state_c = lstm_layer(pose_self_attention, initial_state=initial_state)
    
    return pose_encoder, pose_state_h, pose_state_c



def angle_encoding(angle_input,unit_lstm):
    """
    Encode orientation angles using LSTM.
    """

    angle_self_attention = MultiHeadAttention(num_heads=2, key_dim=16, name='angle_self_attention')(angle_input, angle_input)
    angle_encoder, angle_state_h, angle_state_c = LSTM(unit_lstm, return_sequences=False, return_state=True, name='angle_encoder')(angle_self_attention)
    return angle_encoder, angle_state_h, angle_state_c

def bbox_processing(bbox_input, attention_hidden_state, attention_cell_state, unit_lstm, output_steps=45):
    """
    Encode and decode bounding boxes to predict future bounding boxes, with a longer sequence.
    """
    # Encode the bounding box input sequence (e.g., input 15 bounding boxes)
    bbox_encoder = LSTM(unit_lstm, return_state=True, return_sequences=True, name='bbox_encoder')
    bbox_encoded_output, state_h, state_c = bbox_encoder(bbox_input)

    # Concatenate the LSTM states with the attention states
    context_hidden_state = Concatenate(name='context_h2')([state_h, attention_hidden_state])    
    
    # Transform the concatenated states
    final_hidden_state_transformed = Dense(unit_lstm, activation='tanh', name='context_h_transformed')(context_hidden_state)
    
    # Set the initial states for the decoder LSTM
    initial_state = [final_hidden_state_transformed, final_hidden_state_transformed]
    
    # **Change here**: Use `output_steps` to ensure the decoder predicts 45 time steps
    decoder_input = tf.zeros((tf.shape(bbox_input)[0], output_steps, tf.shape(bbox_input)[-1]))  # Shape: (batch_size, 45, 4)
    
    # Decode bounding boxes (set return_sequences=True for 45 steps)
    bbox_decoder = LSTM(unit_lstm, return_sequences=True, name='bbox_decoder')

    # Predict 45 bounding boxes using the new decoder input
    future_bbox_prediction = bbox_decoder(decoder_input, initial_state=initial_state)
    
    # Apply a time-distributed dense layer to generate the final bounding box coordinates for 45 time steps
    future_bbox_prediction = TimeDistributed(Dense(4, activation='linear'), name='bbox_output')(future_bbox_prediction)
    
    return future_bbox_prediction


def pedestrian_classification(pose_encoder, l2_reg=0.01):
    """
    Classify pedestrian status (standing/walking) with L2 regularization.
    """
    return Dense(2, activation='softmax', kernel_regularizer=l2(l2_reg), name='pedestrian_classification')(pose_encoder)


def film_modulation(action_logits, bbox_input, hidden_units=32, l2_reg=0.0):
    """
    Apply FiLM modulation to bbox_input based on action classification logits.

    Args:
        action_logits: Tensor of shape (batch, num_actions)
        bbox_input: Tensor of shape (batch, seq_len, bbox_feat_dim)
        hidden_units: Number of neurons in FiLM MLP hidden layers
        l2_reg: L2 regularization factor (default=0.0)

    Returns:
        modulated_bbox: Tensor of same shape as bbox_input
    """
    bbox_feat_dim = int(bbox_input.shape[-1])  # Automatically adapt to bbox features

    # MLP to generate FiLM parameters
    x = Dense(hidden_units, activation='relu', kernel_regularizer=l2(l2_reg))(action_logits)
    x = Dense(hidden_units, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    film_params = Dense(2 * bbox_feat_dim, activation='linear', name='film_params')(x)

    # Split into beta and gamma
    beta  = film_params[:, :bbox_feat_dim]
    gamma = film_params[:, bbox_feat_dim:]

    # Reshape for broadcasting across sequence length
    beta  = Reshape((1, bbox_feat_dim))(beta)
    gamma = Reshape((1, bbox_feat_dim))(gamma)

    # FiLM modulation: gamma * x + beta
    modulated_bbox = Add()([Multiply()([bbox_input, gamma]), beta])
    return modulated_bbox


def build_model(unit_lstm,loss_weight):
    """
    Build the model with all components.
    """
    feature_input, pose_input, angle_input, bbox_input = create_inputs()
    print("feature_input.shape: ",feature_input.shape)
    print("pose_input.shape: ",pose_input.shape)
    print("angle_input.shape: ",angle_input.shape)
    print("bbox_input.shape: ",bbox_input.shape)
    _, state_h, state_c = feature_encoding(feature_input,unit_lstm)
    
    # Get the encoded output, hidden state, and cell state from pose_encoding
    pose_encoder, pose_state_h, pose_state_c = pose_encoding(pose_input, state_h, state_c, unit_lstm)
    
    # Get the encoded output, hidden state, and cell state from angle_encoding
    angle_encoder, angle_state_h, angle_state_c = angle_encoding(angle_input, unit_lstm)
    
    # Concatenate both hidden state and cell state
    context_hidden_state = Concatenate(name='context_h')([angle_state_h, pose_state_h])
    context_cell_state = Concatenate(name='context_c')([angle_state_c, pose_state_c, ])


    # Add attention layers
    attention_hidden_state = Attention(name='attention_h')([context_hidden_state, context_hidden_state])
    attention_cell_state = Attention(name='attention_c')([context_cell_state, context_cell_state])
    
    p_s = pedestrian_classification(pose_encoder)
    modulated_bbox = film_modulation(p_s, bbox_input, hidden_units=32)

    future_bbox_prediction = bbox_processing(modulated_bbox, attention_hidden_state, attention_cell_state, unit_lstm, output_steps=45)

    print("future_bbox_prediction:", future_bbox_prediction.shape)


    model = Model(inputs=[feature_input, pose_input, angle_input, bbox_input],
                  outputs=[future_bbox_prediction, p_s])
    model.compile(optimizer=Adam(),
                  loss={'bbox_output': 'mse', 'pedestrian_classification': 'sparse_categorical_crossentropy'},
                  loss_weights={'bbox_output': 1.0, 'pedestrian_classification': loss_weight}
                )

    return model


def train_model(dataset, sequence_length, model, X_feature, X_pose, X_angle, X_bbox, Y_bbox, Y_p_s, unit_lstm, batch_size, loss_weight, epochs, random_state):
    # Define the file name based on the given parameters
    weight_filename = f"{sequence_length}frames_unit_{unit_lstm}_{dataset}.h5"
    
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
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(weight_filename, monitor='val_loss', save_best_only=True, mode='min')

    history = model.fit(
        [X_feature_train, X_pose_train, X_angle_train, X_bbox_train],
        [Y_bbox_train, Y_p_s_train],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([X_feature_val, X_pose_val, X_angle_val, X_bbox_val], [Y_bbox_val, Y_p_s_val]),
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Save the final model weights with the dynamic filename
    final_weight_filename = f"{sequence_length}frames_unit_{unit_lstm}_batch_{batch_size}_loss_{loss_weight}_{dataset}.h5"
    model.save_weights(final_weight_filename)

    # Evaluate the model on the test set
    test_loss = model.evaluate([X_feature_test, X_pose_test, X_angle_test, X_bbox_test], [Y_bbox_test, Y_p_s_test])
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


def evaluate_model_per_sample_center(dataset, seq_lenght, model, X_feature, X_pose, X_angle, X_bbox, Y_bbox, Y_p_s, unit_lstm, batch_size, loss_weight, epochs, random_state):
    weight_filename = f"{seq_lenght}frames_unit_{unit_lstm}_batch_{batch_size}_loss_{loss_weight}_{dataset}.h5"
    
    if os.path.exists(weight_filename):
        print(f"Loading weights from {weight_filename}")
        model.load_weights(weight_filename)
    else:
        print(f"Weight file {weight_filename} not found.")
        return

    total_samples = len(X_feature)
    
    csv_filename = f"bbox_loss_per_sample_{seq_lenght}_unit_{unit_lstm}_batch_{batch_size}_loss_weight_{loss_weight}_{dataset}.csv"
    
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
            writer.writerow(['Sample Index', 'Bbx Loss 0.5s','Bbx Loss 1s','Bbx Loss 1.5s','Bbx Loss 1.5s (evaluate)', 'C_MSE', 'CF_MSE', 'pedestrian_class_loss', 'over_all_loss'])

    total_bbox_loss_1_5_sec = total_bbox_loss_1_sec = total_bbox_loss_0_5_sec = total_C_MSE = total_CF_MSE = total_pedestrian_class_loss =total_overall_loss= 0
    
    for i in range(l_saved, total_samples):
        X_feature_sample = np.expand_dims(X_feature[i], axis=0)
        X_pose_sample = np.expand_dims(X_pose[i], axis=0)
        X_angle_sample = np.expand_dims(X_angle[i], axis=0)
        X_bbox_sample = np.expand_dims(X_bbox[i], axis=0)
        Y_bbox_sample = np.expand_dims(Y_bbox[i], axis=0)
        Y_p_s_sample = np.expand_dims(Y_p_s[i], axis=0)

        ground_truth_centers = calculate_bbox_centers(Y_bbox_sample)
        
        prediction = model.predict([X_feature_sample, X_pose_sample, X_angle_sample, X_bbox_sample], verbose=0)
        predicted_centers = calculate_bbox_centers(prediction[0])

        overall_loss, bbox_loss_1_5_sec_evaluate, pedestrian_class_loss = model.evaluate(
            [X_feature_sample, X_pose_sample, X_angle_sample, X_bbox_sample],
            [Y_bbox_sample, Y_p_s_sample],
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

        total_overall_loss += overall_loss
        total_pedestrian_class_loss += pedestrian_class_loss
        total_C_MSE += C_MSE
        total_CF_MSE += CF_MSE
        total_bbox_loss_1_5_sec += bbox_loss_1_5_sec
        total_bbox_loss_1_sec += bbox_loss_1_sec
        total_bbox_loss_0_5_sec += bbox_loss_0_5_sec

        # Append current metrics to CSV
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i + 1, bbox_loss_0_5_sec, bbox_loss_1_sec, bbox_loss_1_5_sec, bbox_loss_1_5_sec_evaluate, C_MSE, CF_MSE,pedestrian_class_loss,overall_loss])
            
        print(f"{i+1}/{total_samples}")

    average_bbox_loss_1_5_sec = total_bbox_loss_1_5_sec / total_samples
    average_bbox_loss_1_sec = total_bbox_loss_1_sec / total_samples
    average_bbox_loss_0_5_sec = total_bbox_loss_0_5_sec / total_samples
    
    average_C_MSE = total_C_MSE /  total_samples
    average_CF_MSE = total_CF_MSE / total_samples
    average_overall_loss = total_overall_loss / total_samples
    average_pedestrian_class_loss = total_pedestrian_class_loss / total_samples
    
    print(f"average_bbox_loss_1_5_sec: {average_bbox_loss_1_5_sec}")
    print(f"average_bbox_loss_1_sec: {average_bbox_loss_1_sec}")
    print(f"average_bbox_loss_0_5_sec: {average_bbox_loss_0_5_sec}")
    print(f"Average C_MSE: {average_C_MSE}")
    print(f"Average CF_MSE: {average_CF_MSE}")
    print(f"Average pedestrian_class_loss: {average_pedestrian_class_loss}")
    print(f"Average overall_loss: {average_overall_loss}")
    
    return average_bbox_loss_1_5_sec





