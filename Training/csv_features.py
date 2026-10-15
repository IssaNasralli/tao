import pandas as pd
import numpy as np

def normalize_bbxes(bbxes, frame_width, frame_height):
    # Reshape to handle each coordinate
    bbxes = bbxes.astype(np.float64)

    bbxes_reshaped = bbxes.reshape(-1, 16, 4)  # (Batch, TimeSteps, Coordinates)

    # Normalize bounding box coordinates
    bbxes_reshaped[:, :, 0] /= frame_width   # Normalize xmin
    bbxes_reshaped[:, :, 1] /= frame_height  # Normalize ymin
    bbxes_reshaped[:, :, 2] /= frame_width   # Normalize xmax
    bbxes_reshaped[:, :, 3] /= frame_height  # Normalize ymax
    
    # Reshape back to original shape
    bbxes_normalized = bbxes_reshaped.reshape(bbxes.shape)
    return bbxes_normalized

def normalize_poses(poses, frame_width, frame_height):
    # Reshape to split the x and y coordinates
    poses = poses.astype(np.float64)
    poses_reshaped = poses.reshape(-1, 12, 2, 16)  # (Batch, Landmarks, Coordinates, TimeSteps)
    
    # Normalize x and y coordinates
    poses_reshaped[:, :, 0, :] /= frame_width   # Normalize x-coordinates
    poses_reshaped[:, :, 1, :] /= frame_height  # Normalize y-coordinates
    
    # Reshape back to original shape
    poses_normalized = poses_reshaped.reshape(poses.shape)
    return poses_normalized

def denormalize_poses(poses_normalized, frame_width, frame_height):
    # Reverse the normalization process
    poses_reshaped = poses_normalized.reshape(-1, 12, 2, 16)
    poses_reshaped[:, :, 0, :] *= frame_width
    poses_reshaped[:, :, 1, :] *= frame_height
    return poses_reshaped.reshape(poses_normalized.shape)

def denormalize_bbxes(bbxes_normalized, frame_width, frame_height):
    # Reverse the normalization process
    bbxes_reshaped = bbxes_normalized.reshape(-1, 16, 4)
    bbxes_reshaped[:, :, 0] *= frame_width
    bbxes_reshaped[:, :, 1] *= frame_height
    bbxes_reshaped[:, :, 2] *= frame_width
    bbxes_reshaped[:, :, 3] *= frame_height
    return bbxes_reshaped.reshape(bbxes_normalized.shape)

def load_and_preprocess_data(feature_path, angle_path, pose_path, x_traject_path, y_traject_path,feature_max,angle_max, frame_width, frame_height, sequence_length=16):

    # Load the CSV files
    features = pd.read_csv(feature_path, header=None)
    angles = pd.read_csv(angle_path, header=None)
    poses = pd.read_csv(pose_path, header=None)
    x_trajectories = pd.read_csv(x_traject_path, header=None)
    y_trajectories = pd.read_csv(y_traject_path, header=None)

    # Verify that all files have the same number of lines

    print ("len(features:", len(features))
    print ("len(features:", len(angles))
    print ("len(features:", len(poses))
    print ("len(features:", len(x_trajectories))
    print ("len(features:", len(y_trajectories))

    assert len(features) == len(angles) == len(poses) == len(x_trajectories) == len(y_trajectories), "Mismatch in the number of lines across files"

    # Extract action labels
    action_labels = features.iloc[:, -1].values
    mask_standing=action_labels<1
    mask_walking=action_labels>0
    standing=action_labels[mask_standing]
    walking=action_labels[mask_walking]


    # Remove the last column from features, poses, angles, bbxes, and bbxes_future
    features = features.iloc[:, :-1].values
    poses = poses.iloc[:, :-1].values
    angles = angles.iloc[:, :-1].values
    bbxes = x_trajectories.iloc[:, :-1].values
    bbxes_future = y_trajectories.iloc[:, :-1].values


    num_sequences = features.shape[0]
    print("num_sequences", num_sequences)

    # Convert to numpy arrays and reshape
    features = features.reshape(num_sequences, 15, 32)
    poses = poses.reshape(num_sequences, 15, 24)
    angles = angles.reshape(num_sequences, 15, 1)
    bbxes = bbxes.reshape(num_sequences, 15, 4)
    bbxes_future = bbxes_future.reshape(num_sequences, sequence_length, 4)

    print("features.shape: ",features.shape)
    print("poses.shape: ",poses.shape)
    print("angles.shape: ",angles.shape)
    print("bbxes.shape: ",bbxes.shape)
    print("bbxes_future.shape: ",bbxes_future.shape)
    
    return features, poses, angles, bbxes, bbxes_future, action_labels


