import cv2
import mediapipe as mp
import numpy as np
mp_holistic = mp.solutions.holistic

def get_landmark_from_image(image_path):
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        image = cv2.imread(image_path)
        # image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    return None

def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def get_angles(landmarks):
    angles = dict()
    if landmarks.any():
        lshoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value][0],
                     landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value][1]]
        # print(lshoulder)

        lelbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value][0],
                  landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value][1]]

        lwrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value][0],
                  landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value][1]]

        rshoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value][0],
                    landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value][1]]

        relbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value][0],
                 landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value][1]]

        rwrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value][0],
                 landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value][1]]

        left_angle = calc_angle(lshoulder, lelbow, lwrist)
        right_angle = calc_angle(rshoulder, relbow, rwrist)

        angles["left_elbow_angle"] = left_angle
        angles["right_elbow_angle"] = right_angle

    return angles



print(get_landmark_from_image("photo1.jpg"))
photo1_angle = get_angles(get_landmark_from_image("photo1.jpg"))
print(photo1_angle)