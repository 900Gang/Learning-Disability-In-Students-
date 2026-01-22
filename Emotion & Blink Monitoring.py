import os
import cv2
import csv
import time
import numpy as np
import threading
import mediapipe as mp
from datetime import datetime
from collections import deque, Counter

# IMPORTANT: Ensure you use a recent version of Keras (pip install --upgrade keras)
import keras
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Suppress TensorFlow and MediaPipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- CONFIGURATION & INDICES ---
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.22  
CONSEC_FRAMES = 2

# --- STEP 1: ENHANCED LOGGER CLASS ---
class StudentMetricsLogger:
    def __init__(self, filename="student_monitoring_log.csv"):
        self.filename = filename
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Emotion", "Blink_Count", "Stress_Level"])

    def log(self, emotion, blink_count):
        stress_status = "Normal"
        if emotion in ['Sad', 'Angry', 'Fear']:
            stress_status = "Elevated"
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.filename, 'a', newline='') as f:
            csv.writer(f).writerow([timestamp, emotion, blink_count, stress_status])

# --- STEP 2: MULTITHREADED CAMERA ---
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame if self.ret else None

    def stop(self):
        self.stopped = True
        self.cap.release()

# --- STEP 3: INITIALIZATION (FIXED ARCHITECTURE REBUILD) ---

# A. MediaPipe Face Landmarker Fix
model_path = os.path.abspath('face_landmarker.task').replace('\\', '/')
if not os.path.exists(model_path):
    print(f"ERROR: {model_path} not found!")
    exit()

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# B. Manual Emotion Model Construction (Matches your .keras config)
def build_emotion_model():
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
    
    model = Sequential([
        Input(shape=(48, 48, 1)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        
        Conv2D(256, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        
        Conv2D(512, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        
        Conv2D(512, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(7, activation='softmax')
    ])
    return model

print("Rebuilding emotion model structure...")
emotion_model = build_emotion_model()

# Load the weights into the structure we just built
try:
    emotion_model.load_weights("emotion_detector.keras")
    print("Success: Model weights loaded.")
except Exception as e:
    print(f"Weight load failed, trying full model load: {e}")
    emotion_model = keras.models.load_model("emotion_detector.keras")

labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Variables for processing
buffer = deque(maxlen=15)
logger = StudentMetricsLogger()
vs = VideoStream(0).start()

blink_counter = 0
total_blinks = 0

def calculate_ear(landmarks, eye_indices, w, h):
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h_dist = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h_dist)

print("System Active. Press 'ESC' to exit.")

# --- STEP 4: MAIN LOOP ---
while True:
    frame = vs.read()
    if frame is None: continue
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        for landmarks in result.face_landmarks:
            # 1. Blink detection
            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= CONSEC_FRAMES:
                    total_blinks += 1
                blink_counter = 0

            # 2. Emotion analysis
            coords = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
            x_min, y_min = coords.min(axis=0).astype(int)
            x_max, y_max = coords.max(axis=0).astype(int)

            roi = frame[max(0, y_min):min(h, y_max), max(0, x_min):min(w, x_max)]
            stable_emotion = "Analyzing..."
            
            if roi.size > 0:
                gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (48, 48))
                input_data = gray.reshape(1, 48, 48, 1).astype('float32') / 255.0
                pred = emotion_model.predict(input_data, verbose=0)
                buffer.append(labels[np.argmax(pred)])
                stable_emotion = Counter(buffer).most_common(1)[0][0]

            # 3. Logging & UI
            logger.log(stable_emotion, total_blinks)
            
            for pt in coords[::10]:
                cv2.circle(frame, tuple(pt.astype(int)), 1, (0, 255, 0), -1)

            cv2.putText(frame, f"Emotion: {stable_emotion}", (20, 40), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {total_blinks}", (20, 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow("Student Behavior Monitor", frame)
    if cv2.waitKey(1) == 27: break

# --- STEP 5: CLEANUP ---
vs.stop()
cv2.destroyAllWindows()