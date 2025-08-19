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








