import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
import mediapipe as mp
import tensorflow_hub as hub
import gc
import ato
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



def calculate_scale_factors(video_path, desired_width, desired_height):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get original video resolution
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate the scale factors for resizing
    scale_x = desired_width / original_width
    scale_y = desired_height / original_height
    
    # Release the video capture object
    cap.release()
    
    return scale_x, scale_y
        
    
def pop_last_data(pose_data,features_data,teta_data,x_traject_data):
    pose_data.pop(0)
    features_data.pop(0)
    teta_data.pop(0)
    x_traject_data.pop(0)

def get_y_trajectory(boxes,frame_number,scale_x,scale_y,sequence_length):
	y_traject_data=[]
	for i in range(sequence_length):
		matching_boxes = [box for box in boxes if int(box.get("frame")) == frame_number+i+1]
		for box in matching_boxes:
			xtl = int(float(box.get("xtl")) * scale_x)
			ytl = int(float(box.get("ytl")) * scale_y)
			xbr = int(float(box.get("xbr")) * scale_x)
			ybr = int(float(box.get("ybr")) * scale_y)
			y_traject_data.append([xtl, ytl, xbr, ybr])
	return y_traject_data

# function to return the magnitude of a vector
def vec_length(v: np.array):
    return np.sqrt(sum(i ** 2 for i in v))

# function to process a vector parameter and return a normalized vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# function to calculate and return a rotation matrix for quaternion generation
def look_at(eye: np.array, target: np.array):
    axis_z = normalize((eye - target))
    if vec_length(axis_z) == 0:
        axis_z = np.array((0, -1, 0))

    axis_x = np.cross(np.array((0, 0, 1)), axis_z)
    if vec_length(axis_x) == 0:
        axis_x = np.array((1, 0, 0))

    axis_y = np.cross(axis_z, axis_x)
    rot_matrix = np.matrix([axis_x, axis_y, axis_z]).transpose()
    return rot_matrix

def get_angle(pose_landmarks,frame,bbox_coords):
    # Check if pose landmarks are detected
    if pose_landmarks is not None:
        # Get the landmark object for the specified ID
        landmark = pose_landmarks.landmark[11]
        xl = landmark.x
        yl = landmark.y
        zl = landmark.z
        # Get the landmark object for the specified ID
        landmark = pose_landmarks.landmark[12]
        xr = landmark.x
        yr = landmark.y
        zr = landmark.z
        zl,_ =  relative_to_absolute(frame, zl, 0, bbox_coords)
        zr,_ =  relative_to_absolute(frame, zr, 0, bbox_coords)        
        orient = look_at(       np.array(   [xl, yl, zl]   )     ,       np.array(   [xr, yr, zr]  )             )
        vec1 = np.array(orient[0], dtype=float)
        vec3 = np.array(orient[1], dtype=float)
        vec4 = np.array(orient[2], dtype=float)
        # normalize to unit length
        vec1 = vec1 / np.linalg.norm(vec1)
        vec3 = vec3 / np.linalg.norm(vec3)
        vec4 = vec4 / np.linalg.norm(vec4)
    
        M1 = np.zeros((3, 3), dtype=float)  # rotation matrix
    
          # rotation matrix setup
        M1[:, 0] = vec1
        M1[:, 1] = vec3
        M1[:, 2] = vec4
    
        # obtaining the quaternion in cartesian form
        a = np.math.sqrt(np.math.sqrt((float(1) + M1[0, 0] + M1[1, 1] + M1[2, 2]) ** 2)) * 0.5
        b1 = (M1[2, 1] - M1[1, 2]) / (4 * a)
        b2 = (M1[0, 2] - M1[2, 0]) / (4 * a)
        b3 = (M1[1, 0] - M1[0, 1]) / (4 * a)
    
        # converting quaternion to polar form
        A = np.math.sqrt((a ** 2) + (b1 ** 2) + (b2 ** 2) + (b3 ** 2))
        theta = np.math.acos(a / A)
        realAngle = ((np.rad2deg(theta) / 45) - 1) * 180
    
        return realAngle   


from tensorflow.keras.layers import MaxPooling1D, Input
from tensorflow.keras.models import Model
def extract_features(image, model):

    # Preprocess the image.
    image = cv2.resize(image, (224, 224))  # The model expects 224x224 input images.
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize the image to [0, 1].
    image = np.expand_dims(image, axis=0)  # Add batch dimension.

    # Extract features.
    features = model(image)
    
    # Add a pooling layer to downscale the features.
    input_tensor = Input(shape=(features.shape[-1], 1))
    x = MaxPooling1D(pool_size=40)(input_tensor)  # Change to AveragePooling1D if needed.
    pooling_model = Model(inputs=input_tensor, outputs=x)
    
    pooled_features = pooling_model.predict(np.expand_dims(features.numpy(), axis=-1))
    
    return pooled_features.flatten()

def relative_to_absolute(image, x, y, bbox_coords_original):
    x_min, y_min, x_max, y_max = bbox_coords_original
    x = (x_min + (x_max - x_min) * x) / image.shape[1]
    y = (y_min + (y_max - y_min) * y) / image.shape[0]
    return x,y

def visualize_person(image, results, bbox_coords_original,mp_drawing,mp_pose):
    liste = results
    x_min_crop, y_min_crop, x_max_crop, y_max_crop = bbox_coords_original
    if liste is not None:
        for landmark in liste.landmark:
            landmark.x,  landmark.y = relative_to_absolute(image,  landmark.x ,  landmark.y , bbox_coords_original)
        mp_drawing.draw_landmarks(image, liste, mp_pose.POSE_CONNECTIONS)

    return image

def get_landmark_coords_ratio(pose_landmarks, landmark_id):

    # Check if pose landmarks are detected
    if pose_landmarks is not None:
        # Get the landmark object for the specified ID
        landmark = pose_landmarks.landmark[landmark_id]

        # Calculate the x and y pixel coordinates of the landmark
        x = landmark.x
        y = landmark.y

        return x, y
    else:
        # Return None if no landmarks are detected
        return None, None

def predict_video(model_weight, sequence_length, video_folder , checkpoint_file, checkpoint_track):
    
    model_ato = ato.build_model(1024, 1000)
    # Compile the model
    model_ato.compile(
        optimizer=Adam(),
        loss={'bbox_output': 'mse', 'pedestrian_classification': 'sparse_categorical_crossentropy'},
        loss_weights={'bbox_output': 1.0, 'pedestrian_classification': 1000}
    )    
    if os.path.exists(model_weight):
        print(f"Loading weights from {model_weight}")
        model_ato.load_weights(model_weight)
    else:
        print(f"Weight file {model_weight} not found.")
        return

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_videos = set(line.strip() for line in f)
    else:
        processed_videos = set()
          
          
        
    model = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2")                                    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    
    # List video files and sort them by their numeric order
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    video_files_sorted = sorted(video_files, key=lambda x: int(x.split('.')[0]))  # Sorting based on the number before '.mp4'
    
    print (video_files_sorted)   
    with mp_pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        pose_data=[]
        features_data =[]
        teta_data =[]
        y_traject_data =[]
        x_traject_data = []
        frame_count = 0
        current_state = ""
        contradiction=0
        gc.collect()
        for video_file in video_files_sorted:
            gc.collect()
            if video_file.endswith(".mp4"):
                video_path = os.path.join(video_folder, video_file)
                annotation_path = os.path.join(video_folder, video_file.replace(".mp4", ".xml"))
                
                # Check if the video has already been processed
                if video_file in processed_videos:
                    print(f"Skipping {video_file}, already processed.")
                    continue  # Skip to the next video
               
                print(f"Processing {video_path}")
                
                if os.path.exists(checkpoint_track):
                    with open(checkpoint_track, 'r') as f:
                        processed_tracks = set(line.strip() for line in f)
                else:
                    processed_tracks = set()
        
                tree = ET.parse(annotation_path)
                root = tree.getroot()
                track_elements = root.findall(".//track[@label='pedestrian']")
                len_track_elements= len(track_elements)
                
                if track_elements:
                    gc.collect()
                    iterated=0
                    for t in range(len(track_elements)):
                        iterated=iterated+1
                        track_id = f"{video_file}_track_{t}"
                        # Check if the track element has already been processed
                        if track_id in processed_tracks:
                            print(f"Skipping {track_id}, already processed.")
                            continue
                        
                        print(f"Processing {track_id}")
                        
                        pose_data=[]
                        features_data =[]
                        teta_data =[]
                        y_traject_data =[]
                        x_traject_data = []                        
                        pedestrian_track = track_elements[t]
                        boxes = pedestrian_track.findall(".//box")
                        
                        max_frames = 0
                        for box in boxes:
                            frame_number = int(box.get('frame'))
                            if frame_number > max_frames:
                                max_frames = frame_number
                        min_frames=max_frames-len(boxes)
                        max_frames=max_frames-sequence_length
                        cap = cv2.VideoCapture(video_path)
                        desired_width , desired_height = 1200,600
                        #scale_x, scale_y = calculate_scale_factors(video_path, desired_width, desired_height)
                        scale_x, scale_y = 1,1
                        cap.set(cv2.CAP_PROP_POS_FRAMES, min_frames)
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            print(f"{sequence_length}: video_file={video_file}/pedestrian={iterated} of {len_track_elements}/min_frames={min_frames}/frame_number={frame_number}/max_frames={max_frames}")
                            #print("contradiction:", contradiction)
                            
                            
                            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                            #print(f"frame_number & max_frames & min_frames:{frame_number} & {min_frames-1} & {max_frames-1} ") 
                            if (frame_number>max_frames-1):
                                break
                            
                            if (frame_number<min_frames):
                                continue
                            
                            matching_boxes = [box for box in boxes if int(box.get("frame")) == frame_number]
                            for box in matching_boxes:
                                xtl, ytl, xbr, ybr = int(float(box.get("xtl")) * scale_x), int(float(box.get("ytl")) * scale_y), int(float(box.get("xbr")) * scale_x), int(float(box.get("ybr")) * scale_y)
                                current_state = box.find(".//attribute[@name='action']").text.strip().lower()
        
                                c_frame = frame.copy() 
                                cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
                                cv2.putText(frame, f"Ground Truth Action: {current_state}", (xtl, ytl - 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                cv2.putText(frame, f"frame_number: {frame_number}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                xmin, ymin, xmax, ymax = xtl, ytl, xbr, ybr
                                img_np = np.array(c_frame, dtype=np.uint8)
                                width = xmax - xmin
                                height = ymax - ymin
                                
                                width_expansion = int(0.2 * width)
                                height_expansion = int(0.2 * height)
                                
                                xmin2 = max(0, xmin - width_expansion)
                                ymin2 = max(0, ymin - height_expansion)
                                xmax2 = min(img_np.shape[1], xmax + width_expansion)
                                ymax2 = min(img_np.shape[0], ymax + height_expansion)
                                
                                # Extract the sub-image +20% bbx
                                sub_image = img_np[ymin2:ymax2, xmin2:xmax2]
                                # Extract features using Pre-trained CNN
                                features = extract_features(sub_image,model)

                                if sub_image.size > 0:
                                    sub_image_rgb = cv2.cvtColor(sub_image, cv2.COLOR_BGR2RGB)
                                    results = pose.process(sub_image_rgb)


                                    if frame_count < 15:
                                    
                                            if (results.pose_landmarks is not None):

                                                frame = visualize_person(frame, results.pose_landmarks, [xmin2, ymin2, xmax2, ymax2], mp_drawing, mp_pose)                                    
                                                pose_landmarks = results.pose_landmarks
                                                pose_coords = []
                                                for i in range(33):
                                                    if i in [0,1,2,3,4,5,6,7,8,9,10,11,12,17,18,19,20,21,22,23,24]:
                                                        continue
                                                    x, y = get_landmark_coords_ratio(pose_landmarks, i)
                                                    pose_coords.extend([x, y])
                                                teta=get_angle(pose_landmarks,c_frame,[xmin2, ymin2, xmax2, ymax2])    

                                            else:
                                                pose_coords = []
                                                for i in range(33):
                                                    if i in [0,1,2,3,4,5,6,7,8,9,10,11,12,17,18,19,20,21,22,23,24]:
                                                        continue
                                                    x, y = -1,-1
                                                    pose_coords.extend([x, y])
                                                teta=-360
                                            pose_data.append(pose_coords)
                                            teta_data.append(teta)  
                                            features_data.append(features)                                                     
                                            x_traject_data.append([xmin, ymin, xmax, ymax])                                                     
                                            frame_count += 1
                                                                                               
                                    if (frame_count == 15):
                                        y_traject_data= get_y_trajectory(boxes,frame_number,scale_x,scale_y,sequence_length)
                                        if(len(pose_data)==15) and (len(features_data)==15) and (len(teta_data)==15) and (len(y_traject_data)==sequence_length) and (len(x_traject_data)==15):
                                            

                                            # Ensure the data has shape [batch_size, sequence_length, features]

                                            features_data_input = np.expand_dims(features_data, axis=0)  # Add batch dimension if needed
                                            pose_data_input = np.expand_dims(pose_data, axis=0)
                                            teta_data_input = np.expand_dims(teta_data, axis=0)
                                            x_traject_data_input = np.expand_dims(x_traject_data, axis=0)
                                            teta_data_input = np.expand_dims(teta_data_input, axis=-1)
                                            prediction = model_ato.predict([features_data_input, pose_data_input, teta_data_input, x_traject_data_input], verbose=0)
                                            predicted_action = prediction[1] 
                                            print(x_traject_data_input.shape)
                                            if predicted_action[0][0] > 0.5:
                                                predicted_action=f"standing ({predicted_action[0][0]})"
                                            else:
                                                predicted_action=f"walking ({1-predicted_action[0][0]}"
                                                
                                            cv2.putText(frame, f"Predicted Action: {predicted_action}", (xtl, ytl - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 242, 255), 3)

                                            for g in range(sequence_length) :  # Draw only the first 15
                                                    x_min, y_min, x_max, y_max = y_traject_data[g]
                                                    if (g!=sequence_length-1):
                                                        if(g==0) or(g==5) or(g==10) or(g==15) or(g==25) or (g==30) or(g==35) or(g==40):
                                                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green for ground truth boxes
                                                    else:
                                                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue for final ground truth boxes
                                           
                                            predicted_bbox = prediction[0]   
                                            for p in range(sequence_length):
                                                bbox = predicted_bbox[0, p, :]
                                                x_min, y_min, x_max, y_max = bbox
                                                x_min = int(x_min)
                                                y_min = int(y_min)
                                                x_max = int(x_max)
                                                y_max = int(y_max)
                                                if (p!=sequence_length-1):
                                                    if(p==0) or(p==5) or(p==10) or(p==15) or(p==25) or (p==30) or(p==35) or(p==40):
                                                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 242, 255), 1) # Yellow for  predicted boxes
                                                else:
                                                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1) # Red for  predicted boxes
                                                                                        
                                          
                                            frame_count = frame_count - 1
                                            pop_last_data(pose_data,features_data,teta_data,x_traject_data)
                                            print(f"15 feature collected and {sequence_length} future trajectories collected")
                                        else:
                                            pose_data=[]
                                            features_data =[]
                                            teta_data =[]
                                            y_traject_data =[]
                                            x_traject_data = []
                                            frame_count=0
                            # Define the path to save the frames
                            output_dir = f"{sequence_length}f"
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            
                            frame = cv2.resize(frame, (desired_width, desired_height))
                            cv2.imshow('Frame', frame)


                            del frame
                            gc.collect()

                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
        
                        cap.release()
                        cv2.destroyAllWindows()
                        with open(checkpoint_track, 'a') as f:
                            f.write(f"{track_id}\n")
                        print(f"{track_id} processed and added to the checkpoint.")
                
                # After processing, add the video file to the checkpoint file
                with open(checkpoint_file, 'a') as f:
                    f.write(f"{video_file}\n")
                print(f"{video_file} processed and added to the checkpoint.")
     
                if os.path.exists(f"checkpoint_track_{sequence_length}.txt"):
                    os.remove(f"checkpoint_track_{sequence_length}.txt")
    
                gc.collect()
                


dataset_folder = "JAAD"
model_weight = "weight.h5"

# Process videos and annotations
sequence_length=45
predict_video(model_weight, sequence_length, dataset_folder, checkpoint_file=f"checkpoint_file_{sequence_length}.txt", checkpoint_track=f"checkpoint_track_{sequence_length}.txt")