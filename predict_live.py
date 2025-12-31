import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load TFLite model

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    flip_img=cv2.flip(img_rgb,1)
    results = hands.process( flip_img)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 21 landmarks × (x,y,z)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks, dtype=np.float32).flatten().reshape(1, 63)

            # Predict
            interpreter.set_tensor(input_details[0]['index'], landmarks)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]

            predicted_class = np.argmax(prediction)
            if(predicted_class==0):
                output="A"
            elif(predicted_class==1):
                output="B"
            elif(predicted_class==2):
                output="C"
            elif(predicted_class==3):
                output="D"
            else:
                output="E"
            
            cv2.putText(frame, f"Prediction: {output}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Live Hand Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()