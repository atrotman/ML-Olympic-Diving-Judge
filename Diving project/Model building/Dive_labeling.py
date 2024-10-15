import os
import glob
import cv2
import numpy as np
import json
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor

### Step 1: Extract Keyframes from Videos ###

def extract_frames(video_path, output_folder, frame_interval=2):
    """
    Extracts frames from a video at a set interval and saves them
    
    Parameters:
    video_path: Path to the input video file
    output_folder: Path to the folder where the extracted frames will be saved
    frame_interval: Interval frames are extracted

    Returns:
    frames: List of extracted frames as numpy arrays
    """
    frames = []

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return frames

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:  # Exit the loop when no more frames
            break

        # Save frame every frame_interval frames
        if frame_number % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_number}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame)

        frame_number += 1

    cap.release()
    return frames


### Step 2: Stabilize Frames to Remove Camera Movement ###

def stabilize_frames(frames):
    """
    Remove the effect of camera motion to stabalize frames

    Parameters:
    frames: List of frames

    Returns:
    stabilized_frames: List of stabilized frames
    """
    stabilized_frames = []

    # Use first frame as reference frame
    prev_frame = frames[0]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    stabilized_frames.append(prev_frame)

    for i in range(1, len(frames)):
        current_frame = frames[i]
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (feature-based motion estimation)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Create a coordinate grid for the image
        h, w = prev_gray.shape
        flow_map = np.meshgrid(np.arange(w), np.arange(h))
        flow_map = np.array(flow_map).transpose(1, 2, 0).astype(np.float32)

        # Add the flow to the coordinate grid
        flow_map += flow

        # Warp the current frame to align with the previous one
        stabilized_frame = cv2.remap(current_frame, flow_map, None, cv2.INTER_LINEAR)

        stabilized_frames.append(stabilized_frame)
        prev_gray = current_gray

    return stabilized_frames


### Step 3: Apply Pose Estimation to Extract Body Positions ###

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def process_frame_with_pose_estimation(image):
    """
    Processes an image to detect the pose (body joints) using MediaPipe and returns the pose landmarks
    
    Parameters:
    image: Image frame
    display: If True, displays the image with pose landmarks
    
    Returns:
    pose_landmarks: Dictionary of detected pose landmarks
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(image_rgb)

    # Collect detected pose landmarks (joints)
    landmarks = {}
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmark_name = mp_pose.PoseLandmark(idx).name
            landmarks[landmark_name] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }

    return landmarks


### Step 4: Detect Rotations and Twists ###

def detect_rotations_and_twists(pose_data):
    """
    Detects the number of rotations and twists by analyzing changes in key joint positions

    Parameters:
    pose_data : List of pose landmarks data for each frame (dict of joint positions)

    Returns:
    rotation_count: Number of rotations detected
    twist_count: Number of twists detected
    """
    previous_angle = None
    rotation_count = 0
    twist_count = 0

    for frame_pose in pose_data:
        # Ensure that left hip and right hip are detected in the frame
        if 'LEFT_HIP' in frame_pose and 'RIGHT_HIP' in frame_pose:
            left_hip = (frame_pose['LEFT_HIP']['x'], frame_pose['LEFT_HIP']['y'])
            right_hip = (frame_pose['RIGHT_HIP']['x'], frame_pose['RIGHT_HIP']['y'])

            # Calculate the angle between the hips for rotation
            current_angle = calculate_angle_between_joints(left_hip, right_hip)

            # Count a rotation if there's a significant angle change
            if previous_angle is not None:
                angle_diff = abs(current_angle - previous_angle)
                if angle_diff > 180:
                    rotation_count += 1

            previous_angle = current_angle

        # Ensure that left shoulder and right shoulder are detected in the frame for twist detection
        if 'LEFT_SHOULDER' in frame_pose and 'RIGHT_SHOULDER' in frame_pose:
            left_shoulder = (frame_pose['LEFT_SHOULDER']['x'], frame_pose['LEFT_SHOULDER']['y'])
            right_shoulder = (frame_pose['RIGHT_SHOULDER']['x'], frame_pose['RIGHT_SHOULDER']['y'])

            # Calculate the angle between the shoulders for twist
            shoulder_angle = calculate_angle_between_joints(left_shoulder, right_shoulder)

            # Count a twist if there's a significant angle change relative to the hips
            if previous_angle is not None:
                twist_diff = abs(shoulder_angle - previous_angle)
                if twist_diff > 45:
                    twist_count += 1

    return rotation_count, twist_count


def calculate_angle_between_joints(joint1, joint2):
    """
    Calculates the angle between two joints (x, y coordinates) in degrees
    
    Parameters:
    joint1, joint2: Tuple representing the (x, y) coordinates of two joints
    
    Returns:
    Angle in degrees between the two joints
    """
    return np.arctan2(joint2[1] - joint1[1], joint2[0] - joint1[0]) * 180 / np.pi


### Step 5: Detect Splash Size ###

def detect_splash(frame_before_entry, entry_frame):
    """
    Detects the size of the splash by comparing the frame before water entry and the entry frame

    Parameters:
    frame_before_entry: Frame before the diver enters the water
    entry_frame: Frame where the diver enters the water

    Returns:
    splash_size: Size of the splash in terms of width and height
    """
    # Convert frames to grayscale
    gray_before_entry = cv2.cvtColor(frame_before_entry, cv2.COLOR_BGR2GRAY)
    gray_entry = cv2.cvtColor(entry_frame, cv2.COLOR_BGR2GRAY)

    # Get the absolute difference between frames
    diff = cv2.absdiff(gray_before_entry, gray_entry)

    # Apply a threshold to isolate the splash area
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # Find contours (i.e., the splash region)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    splash_size = (0, 0)

    # Draw bounding box around the largest contour (splash)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        splash_size = (w, h)

    return splash_size


### Step 6: Get the Difficulty of the Dive###

# Diificulty dictionary with by dive and position
dive_types = {
    "Inward 3 1/2 Somersaults": {
        "Tuck": 3.2,
        "Pike": 3.5
    },
    "Reverse 3 1/2 Somersaults": {
        "Tuck": 3.4
    },
    "Back 3 1/2 Somersaults": {
        "Tuck": 3.3,
        "Pike": 3.6
    },
    "Armstand Back 2 Somersaults 2 1/2 Twists": {
        "Free": 3.6
    },
    "Armstand Back 3 Somersaults": {
        "Tuck": 3.3,
        "Pike": 3.5
    },
    "Armstand Forward 3 Somersaults": {
        "Pike": 3.3
    },
    "Forward 4 1/2 Somersault": {
        "Tuck": 3.7,
        "Pike": 4.1
    },
    "Forward 3 1/2 Somersaults 1 Twist": {
        "Pike": 3.6
    },
    "Forward 2 1/2 Somersaults 3 Twists": {
        "Pike": 3.8
    },
    "Back 2 1/2 Somersaults 2 1/2 Twists": {
        "Pike": 3.6
    },
    "Back 2 1/2 Somersaults 1 1/2 Twists": {
        "Pike": 3.2
    }
}

def get_difficulty(dive_type, position):
    """
    Gets the difficulty of a dive based on the dive type and position
    
    Parameters:
    dive_type: Type of dive
    position: Position of the dive
    
    Returns:
    Degree of Difficulty: The difficulty score associated with the dive type and position
    """
    # Normalize strings to handle case and space differences
    normalized_dive_type = dive_type.strip().lower()
    normalized_position = position.strip().lower()
    # Normalize the dictionary keys and positions
    normalized_dive_types = {key.lower(): {pos.lower(): value for pos, value in positions.items()} for key, positions in dive_types.items()}

    if normalized_dive_type in normalized_dive_types:
        if normalized_position in normalized_dive_types[normalized_dive_type]:
            return normalized_dive_types[normalized_dive_type][normalized_position]
    return 0


### Step 7: Load Dive Labels from JSON ###

def get_dive_info_from_json(json_file, video_filename):
    """
    Looks up the dive type, position, and score for a given video filename from a JSON file
    
    Parameters:
    json_file: Path to the JSON file with dive labels
    video_filename: The video filename
    
    Returns:
    dive_type: The dive type from the JSON
    position: The dive position from the JSON
    score: The score of the dive from the JSON
    """
    with open(json_file, 'r') as file:
        data = json.load(file)

        # Loop through the keys and find a matching filename
        for json_video_filename in data:
            if os.path.basename(json_video_filename) == video_filename:
                dive_info = data[json_video_filename]
                dive_type = dive_info.get('dive_type', 'Unknown')
                position = dive_info.get('position', 'Unknown')
                score = dive_info.get('score', 0.0)
                return dive_type, position, score
    return "Unknown", "Unknown", 0.0 # Return default if not found


### Step 8: Automate the Entire Pipeline for Multiple Videos ###

def automate_dive_annotation(video_path, output_folder, dive_type, position, score):
    """
    Automates the full pipeline for diving annotation: extracts frames, detects poses, rotations, splash size, and difficulty
    
    Parameters:
    video_path: Path to the video file
    output_folder: Path to the output folder
    dive_type: Type of dive
    position: Position of the dive
    score: Score of the dive

    Returns:
    results: Dictionary containing the labeled results
    """
    # Step 1: Extract frames from video
    frames = extract_frames(video_path, output_folder)
    
    # Step 2: Stabilize frames
    stabilized_frames = stabilize_frames(frames)
    
    # Step 3: Loop over stabilized frames and apply pose detection
    pose_data = [process_frame_with_pose_estimation(frame) for frame in stabilized_frames]
    
    # Step 4: Detect rotations and twists
    rotations, twists = detect_rotations_and_twists(pose_data)
    
    # Step 5: Detect splash size
    if len(frames) >= 2:
        splash_size = detect_splash(frames[-2], frames[-1])
    else:
        splash_size = (0, 0)
    
    # Step 6: Get difficulty based on dive type and position
    difficulty = get_difficulty(dive_type, position)
    
    # Compile results
    results = {
        'video': os.path.basename(video_path),
        'dive_type': dive_type,
        'position': position,
        'difficulty': difficulty,
        'rotations': rotations,
        'twists': twists,
        'splash_size': splash_size,
        'score': score
    }
    return results


def process_videos_in_folder(input_folder, output_folder, json_file, output_json_path):
    """
    Processes all videos in a folder, applies dive annotation, and saves results to a JSON file
    
    Parameters:
    input_folder: Path to the folder containing input videos
    output_folder: Path to the output folder
    json_file: Path to the JSON file containing dive information
    output_json_path: Path to the output JSON file to save the results
    """
    # Get a list of all video files in the input folder
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    all_results = []

    # Use ThreadPoolExecutor to process multiple videos concurrently
    with ThreadPoolExecutor() as executor:
        future_to_video = {
            executor.submit(automate_dive_annotation, video_file, os.path.join(output_folder, os.path.basename(video_file).split('.')[0]), *get_dive_info_from_json(json_file, os.path.basename(video_file))): video_file
            for video_file in video_files
        }

        # Collect results as each future completes
        for future in future_to_video:
            video_file = future_to_video[future]
            try:
                result = future.result()
                # Print the progress of each video processing
                print(f"Processing video: {video_file} with dive type: {result['dive_type']}, position: {result['position']}, score: {result['score']}")
                all_results.append(result)
            except Exception as exc:
                # Handle exceptions that occur during video processing
                print(f"{video_file} generated an exception: {exc}")

    # Results to output JSON file
    with open(output_json_path, 'w') as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    process_videos_in_folder("Shortened videos/videos", "Shortened videos/Frames and labeling/Extracted frames", json_file="Shortened videos/olympic_dive_types.json", output_json_path="Shortened videos/Output data/labeled_results.json")