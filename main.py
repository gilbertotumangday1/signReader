# import libraries
import cv2
import mediapipe as mp
import torch
import numpy as np
import pickle
from training.modeltrainer import MLP

#load model and encoder
input_size = 63
#encoder
with open("resources/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
#model
model = MLP(input_size=input_size, num_classes=len(encoder.classes_))
model.load_state_dict(torch.load("resources/asl_mlp_model.pth"))
model.eval()

#mediapipe stuff
hand_detector = mp.solutions.hands
hand_instance = hand_detector.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.7, min_tracking_confidence=0.5
)
hand_drawing = mp.solutions.drawing_utils

#setup webcam
webcam = cv2.VideoCapture(0)

#display loop
while True:
    success, frame = webcam.read()
    if not success:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #get the resulting hand data
    result = hand_instance.process(rgb_frame)

    if result.multi_hand_landmarks:
        #for each landmark of the hand
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            hand_drawing.draw_landmarks(frame, hand_landmarks, hand_detector.HAND_CONNECTIONS)

            # Extract and normalize landmarks relative to the wrist
            raw_landmarks = [pt for pt in hand_landmarks.landmark]
            wrist = raw_landmarks[0]
            normalized = []
                
            #normalize the location (get rid of where the hand is located)
            for pt in raw_landmarks:
                x = pt.x - wrist.x
                y = pt.y - wrist.y
                z = pt.z - wrist.z
                normalized.extend([x, y, z])

            # Convert to tensor
            landmarks_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)

            # Predict
            with torch.no_grad():
                output = model(landmarks_tensor)
                prediction = torch.argmax(output, dim=1).item()
                label = encoder.inverse_transform([prediction])[0]

            # Show prediction on frame
            cv2.putText(frame, f"Predicted: {label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Display window
    cv2.imshow("Hand Tracking (press q to quit)", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()
