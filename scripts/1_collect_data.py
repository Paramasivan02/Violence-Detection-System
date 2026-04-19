
import cv2
import mediapipe as mp
import csv
import os
import numpy as np

def extract_pose_landmarks(video_path, label, output_csv):
    """
    Extracts pose landmarks from a video file and saves them to a CSV.

    Args:
        video_path (str): Path to the input video file.
        label (str): The label for the video (e.g., 'normal' or 'abnormal').
        output_csv (str): Path to the output CSV file.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Check if the CSV file needs a header
    write_header = not os.path.exists(output_csv)

    with open(output_csv, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Define header
        header = ['label']
        for i in range(33):
            header += [f'x_{i}', f'y_{i}']
        
        if write_header:
            csv_writer.writerow(header)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image and find landmarks
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark
                
                # Flatten the landmarks into a single row
                row = [label]
                for landmark in landmarks:
                    row.extend([landmark.x, landmark.y])
                
                # Write the row to the CSV
                csv_writer.writerow(row)
                frame_count += 1

    print(f"Processed {frame_count} frames from {video_path} and saved to {output_csv}")
    
    cap.release()
    pose.close()

if __name__ == '__main__':
    # --- CONFIGURATION ---
    # Create the main data directory if it doesn't exist
    DATA_DIR = '../data'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    OUTPUT_CSV = os.path.join(DATA_DIR, 'pose_data.csv')
    
    # --- INSTRUCTIONS ---
    # 1. Create folders `data/raw_videos/normal` and `data/raw_videos/abnormal`.
    # 2. Place your video files in the respective folders.
    # 3. The script will automatically process them.

    video_base_path = os.path.join(DATA_DIR, 'raw_videos')
    categories = {'normal': 0, 'abnormal': 1}

    # Automatically process all videos in the specified directories
    for category, label_id in categories.items():
        category_path = os.path.join(video_base_path, category)
        if not os.path.exists(category_path):
            print(f"Warning: Directory not found, skipping: {category_path}")
            continue
            
        for video_file in os.listdir(category_path):
            video_path = os.path.join(category_path, video_file)
            print(f"Processing video: {video_path} with label: {category} ({label_id})")
            extract_pose_landmarks(video_path, label_id, OUTPUT_CSV)

    print("Data collection finished!")

