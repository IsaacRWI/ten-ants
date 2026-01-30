import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 854)
cap.set(4, 480)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        # print(results.face_landmarks)
        # print(results.pose_landmarks)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.namedWindow("live camera")
        # cv2.resizeWindow("live camera", 1920, 1080)
        cv2.imshow("live camera", cv2.flip(image, 1))

        if cv2.waitKey(10) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
