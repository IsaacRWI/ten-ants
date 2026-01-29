import cv2
import mediapipe as mp
import numpy as np

def get_landmark_from_image(image_path):
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        image = cv2.imread(image_path)
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        if results.pose_landmark:
            landmarks = results.pose_landmarks.landmark
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    return None
