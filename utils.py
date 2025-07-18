import cv2
import os
import csv
import mediapipe as mp
from datetime import datetime

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


hands = mp_hands.Hands(max_num_hands=2)
def extract_features(hand_img):
    hand_img = cv2.resize(hand_img, (64, 64))
    features = hand_img.flatten()
    return features


data_dir = "sign_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


cap = cv2.VideoCapture(0)
print("📷 Press A-Z or 0-9 to label and save hand data. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

   
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    
    cv2.imshow("Sign Language Capture", frame)
    def normalize_landmarks(landmarks):
        wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
        normalized = []
        for i in range(0, len(landmarks), 3):
            normalized.append(landmarks[i] - wrist_x)    
            normalized.append(landmarks[i+1] - wrist_y)   
            normalized.append(landmarks[i+2] - wrist_z)   
        return normalized

    key = cv2.waitKey(1) & 0xFF  
    if key != 255:
        print("🔤 Key pressed:", key)

  
    if key == ord('q'):
        break


    if (48 <= key <= 57) or (65 <= key <= 90) or (97 <= key <= 122):
        label = chr(key).upper()
        print(f"⏳ Get ready to show '{label}' sign... (2s)")
        cv2.putText(frame, f"Get ready for '{label}'", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Sign Language Capture", frame)
        cv2.waitKey(2000) 

        
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            all_coords = []
            for hand_landmarks in result.multi_hand_landmarks:
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                all_coords.append(coords)

            flat_coords = [val for coords in all_coords for val in coords]

            label_folder = os.path.join(data_dir, label)
            os.makedirs(label_folder, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            filename = f"{label}_{timestamp}.csv"
            file_path = os.path.join(label_folder, filename)

            with open(file_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(flat_coords)

            print(f"✅ Saved {len(all_coords)} hand(s) for '{label}' as {filename}")
        else:
            print("❌ No hand detected. Try again.")

cap.release()
cv2.destroyAllWindows()



