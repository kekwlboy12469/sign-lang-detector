import cv2
import mediapipe as mp
import joblib
import numpy as np


model = joblib.load('models/sign_model_63.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
scaler = joblib.load('models/scaler.pkl')


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)


cap = cv2.VideoCapture(0)

print("[INFO] Starting live prediction... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction = "No hand detected"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
          
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
        
            input_data = np.array(landmarks).reshape(1, -1)
            input_scaled = scaler.transform(input_data)

       
            y_pred = model.predict(input_scaled)
            try:
                predicted_label = label_encoder.inverse_transform(y_pred)[0]
                prediction = f"Prediction: {predicted_label}"
            except ValueError:
                prediction = "Unknown sign (unseen label)"

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Sign Language Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
