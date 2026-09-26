import cv2
import numpy as np
import torch
import os
from collections import deque
from model import ActionRecognitionSTGCN
import mediapipe as mp


DATASET_TYPE = "UCF101"  # Επιλογές: "UCF101" ή "HMDB51"
INPUT_VIDEO = "Man jumping jack.mp4"  
OUTPUT_VIDEO = "Jumping_jack_result_ucf101.mp4" 

if DATASET_TYPE == "UCF101":
    NUM_CLASSES = 101
    MODEL_WEIGHTS = "best_model_ucf101_official.pth"
    DATA_DIR = r"E:\Thesis_project\Human_action_recognition_opencv\processed_data"
else:
    NUM_CLASSES = 51
    MODEL_WEIGHTS = "best_model_hmdb51_official.pth"
    DATA_DIR = r"E:\Thesis_project\Human_action_recognition_opencv\processed_data_hmdb51"

FIXED_FRAMES = 150

# ==========================================
# ΠΡΟΕΤΟΙΜΑΣΙΑ
# ==========================================
classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading {DATASET_TYPE} model on {device}...")

model = ActionRecognitionSTGCN(in_channels=8, num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
model.eval()

# Αρχικοποίηση MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def preprocess_skeleton_data(skeleton_buffer):
    data = np.array(skeleton_buffer)
    left_hip = data[:, 23, :3]
    right_hip = data[:, 24, :3]
    center = (left_hip + right_hip) / 2.0 
    center = center[:, np.newaxis, :]
    data[:, :, :3] = data[:, :, :3] - center
    velocity = np.zeros_like(data)
    velocity[1:] = data[1:] - data[:-1]
    data = np.concatenate((data, velocity), axis=-1)
    data = data.reshape(FIXED_FRAMES, -1)
    return torch.FloatTensor(data).unsqueeze(0)

# ==========================================
# ΕΠΕΞΕΡΓΑΣΙΑ ΒΙΝΤΕΟ
# ==========================================
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"Σφάλμα: Δεν βρέθηκε το βίντεο '{INPUT_VIDEO}'. Βεβαιώσου ότι το έβαλες στον ίδιο φάκελο!")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

skeleton_buffer = deque(maxlen=FIXED_FRAMES)
current_prediction = "Waiting for data..."
confidence = 0.0

print(f"Processing video '{INPUT_VIDEO}'. This may take a moment...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    frame_data = np.zeros((33, 4))
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for i, lm in enumerate(results.pose_landmarks.landmark):
            frame_data[i] = [lm.x, lm.y, lm.z, lm.visibility]
            
    skeleton_buffer.append(frame_data)
    
    if len(skeleton_buffer) > 0:
        padded_buffer = list(skeleton_buffer)
        while len(padded_buffer) < FIXED_FRAMES:
            padded_buffer.append(padded_buffer[-1])
            
        input_tensor = preprocess_skeleton_data(padded_buffer).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_prob, predicted_idx = torch.max(probabilities, 1)
            
            if max_prob.item() > 0.1:
                current_prediction = classes[predicted_idx.item()]
                confidence = max_prob.item() * 100

    cv2.rectangle(frame, (10, 10), (700, 70), (0, 0, 0), -1)
    text = f"Action: {current_prediction} ({confidence:.1f}%)"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    out.write(frame)

cap.release()
out.release()
print(f"Ολοκληρώθηκε! Το αποτέλεσμα αποθηκεύτηκε στο: {OUTPUT_VIDEO}")