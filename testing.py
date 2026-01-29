import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        # print(results.face_landmarks)
        # print(results.pose_landmarks)

        cv2.imshow("camera", cv2.flip(frame, 1))

        if cv2.waitKey(10) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
