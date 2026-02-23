import time
import mediapipe as mp
import cv2
from posture_to_data import get_hand_state, get_arms_state, get_elbow_angles, get_mouth_state
from tracking_to_image import state_to_meme
import os

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  #854
cap.set(4, 720)  #480
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
        mouth_state = get_mouth_state(results)

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        hand_state = get_hand_state(results.right_hand_landmarks)
        # print(results.right_hand_landmarks)
        print(hand_state)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        arm_state = get_arms_state(results.pose_landmarks)
        print(get_elbow_angles(results.pose_landmarks))
        print(arm_state)
        print(mouth_state)
        # time.sleep(0.01)
        # os.system("cls")

        cv2.namedWindow("live camera")
        # cv2.resizeWindow("live camera", 854, 480)
        # cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        image = cv2.flip(image,1)
        meme = state_to_meme(arm_state, hand_state, mouth_state)
        if meme:
            overlay = cv2.resize(cv2.imread(meme), (900, 720))  #(x, y)
            image[0:720, 190:1090] = overlay  # (y, x)
        cv2.putText(image, f"state: {arm_state}, {hand_state}, {mouth_state}", (10, 30), cv2.QT_FONT_NORMAL, 0.8, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("live camera", image)
        # cv2.imshow("live camera", image)

        if cv2.waitKey(10) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
