import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import gc

RAW_DIR = r"E:\μεταπτυχιακό Γιάννης\μαθήματα ΜΠΣ\Διπλωματική\Code\Human_action_recognition_opencv\raw_videos\hmdb51"
PROCESSED_DIR = r"E:\μεταπτυχιακό Γιάννης\μαθήματα ΜΠΣ\Διπλωματική\Code\Human_action_recognition_opencv\processed_data_hmdb51"
MODEL_PATH = 'pose_landmarker_heavy.task'

def extract_skeleton(video_path):
    # Initialize the model inside the function so it resets for every video
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    skeletons = []
    frame_index = 0
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Timestamp starts at 0 for every new video
            timestamp_ms = int(frame_index * (1000 / 30))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if result.pose_landmarks:
                # Extract the 4th dimension for occlusion awareness
                joints = [[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in result.pose_landmarks[0]]
                skeletons.append(joints)
            else:
                # Pad with zeros shaped (33, 4) instead of (33, 3)
                skeletons.append(np.zeros((33, 4)).tolist())
                
            frame_index += 1
            
    cap.release()
    return np.array(skeletons)

def process_dataset():
    classes = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    total_videos = sum([len([v for v in os.listdir(os.path.join(RAW_DIR, c)) if v.endswith('.avi')]) for c in classes])
    processed_videos = 0
    skipped_videos = 0
    
    print(f"Starting paced extraction for {total_videos} videos...\n")
    
    for action_class in classes:
        class_raw_dir = os.path.join(RAW_DIR, action_class)
        class_processed_dir = os.path.join(PROCESSED_DIR, action_class)
        
        os.makedirs(class_processed_dir, exist_ok=True)
        videos = [v for v in os.listdir(class_raw_dir) if v.endswith('.avi')]
        
        for video in videos:
            raw_video_path = os.path.join(class_raw_dir, video)
            save_path = os.path.join(class_processed_dir, video.replace('.avi', '.npy'))
            
            # Skip check για να μήν κάνουμε απο την αρχή όλα τα δεδομένα (σε περίπτωση που παγώσει η επεξεργασία)
            if os.path.exists(save_path):
                skipped_videos += 1
                continue
                
            try:
                print(f"Extracting [{processed_videos + skipped_videos + 1}/{total_videos}]: {action_class} / {video}")
                skeleton_data = extract_skeleton(raw_video_path)
                
                if skeleton_data.size > 0:
                    np.save(save_path, skeleton_data)
                    processed_videos += 1
                
                # καθυστέρηση για να μην κάνει overload ο υπολογιστής
                time.sleep(0.5)
                
                # Memory management: αδείαζει την ram μετά απο κάθε 50 βίντεο
                if processed_videos > 0 and processed_videos % 50 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"Failed to process {video}: {str(e)}. Moving to next.")
                
    print(f"\nProcessing Complete. {processed_videos} newly extracted, {skipped_videos} previously skipped.")

if __name__ == "__main__":
    process_dataset()