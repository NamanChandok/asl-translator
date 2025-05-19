import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

model = load_model("asl_cnn_model.keras")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(1)
labels = list("0123456789abcdefghijklmnopqrstuvxyz")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

def extract_hand(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]


        landmarks = []
        for landmark in hand_landmarks.landmark:
            h, w, _ = frame.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            landmarks.append([x, y])

        landmarks = np.array(landmarks)
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)

        padding = 30
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)

        hand_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        hull = cv2.convexHull(landmarks)
        cv2.fillConvexPoly(hand_mask, hull, 255)

        kernel = np.ones((7, 7), np.uint8)
        hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)

        hand_on_black = np.zeros_like(frame)
        hand_on_black[hand_mask > 0] = frame[hand_mask > 0]

        hand_roi = hand_on_black[y_min:y_max, x_min:x_max]

        if hand_roi.size == 0:
            return None, frame, None, None

        lab = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_hand_roi = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_hand_roi, frame, (x_min, y_min, x_max, y_max), hand_mask

    return None, frame, None, None

def preprocess_for_model(hand_roi):
    roi_resized = cv2.resize(hand_roi, (64, 64))

    roi_normalized = roi_resized.astype("float32") / 255.0

    roi_reshaped = np.expand_dims(roi_normalized, axis=0)

    return roi_reshaped

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    hand_roi, annotated_frame, bbox, hand_mask = extract_hand(frame)

    if hand_roi is not None and bbox is not None:
        try:
            roi_preprocessed = preprocess_for_model(hand_roi)
            prediction = model.predict(roi_preprocessed, verbose=0)
            pred_index = np.argmax(prediction)
            pred_label = labels[pred_index]
            confidence = prediction[0][pred_index]

            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'{pred_label.upper()} ({confidence:.2f})',
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            display_hand = cv2.resize(hand_roi, (160, 160))
            h, w = display_hand.shape[:2]
            annotated_frame[10:10+h, 10:10+w] = display_hand

        except Exception as e:
            print(f"Processing error: {e}")
    else:
        cv2.putText(annotated_frame, "No hand detected", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Recognition", annotated_frame)

    if hand_roi is not None:
        cv2.imshow("Hand Isolation (Model Input)", hand_roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
