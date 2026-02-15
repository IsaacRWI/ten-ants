import cv2
import mediapipe as mp
import numpy as np
mp_holistic = mp.solutions.holistic
from photos import *

def get_landmark_from_image(image_path):
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        image = cv2.imread(image_path)
        # image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        if results.pose_landmarks:
            # landmarks = results.pose_landmarks.landmark
            # return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            return results.pose_landmarks
    return None

def get_hand_landmarks_from_image(image_path):
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        image = cv2.imread(image_path)
        # image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        # print(results.right_hand_landmarks)
        #hand_landmarks = []

        # if results.left_hand_landmarks:
            # left_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
            # hand_landmarks["left_hand"] = left_hand_landmarks
            # hand_landmarks.append(results.left_hand_landmarks)
        # if results.right_hand_landmarks:
            # right_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
            # hand_landmarks["right_hand"] = right_hand_landmarks
            # hand_landmarks.append(results.right_hand_landmarks)
        return results.right_hand_landmarks

def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def get_elbow_angles(landmarks):
    angles = dict()
    if landmarks:
        lshoulder = [landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
        # print(lshoulder)

        lelbow = [landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]

        lwrist = [landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]

        rshoulder = [landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]

        relbow = [landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]

        rwrist = [landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]

        left_angle = calc_angle(lshoulder, lelbow, lwrist)
        right_angle = calc_angle(rshoulder, relbow, rwrist)

        angles["left_elbow_angle"] = left_angle  # left hand actually corresponds to your left hand in person not on camera
        angles["right_elbow_angle"] = right_angle

    return angles

def get_hand_state(landmarks):
    if landmarks:
        # print("dict is not empty")
        thumb_tip = landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP.value].y
        index_tip = landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP.value].y
        middle_tip = landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].y
        ring_tip = landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP.value].y
        pinky_tip = landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP.value].y
        # print(thumb_tip)
        # print(index_tip)
        # print(middle_tip)
        # print(ring_tip)
        # print(pinky_tip)

        if thumb_tip < index_tip < middle_tip < ring_tip < pinky_tip:
            return "thumbs_up"
        elif index_tip < thumb_tip < ring_tip < pinky_tip and middle_tip < thumb_tip < ring_tip < pinky_tip:
            return "peace"
        elif (index_tip < thumb_tip and
            index_tip < middle_tip and
            index_tip < ring_tip and
            index_tip < pinky_tip):
            return "pointing"
        elif (thumb_tip > index_tip > middle_tip and
            middle_tip < ring_tip < pinky_tip):
            return "open"
        elif thumb_tip > index_tip > middle_tip > ring_tip > pinky_tip:
            return "thumbs_down"
        return "unknown pose"
    else: return "error"

def get_arms_state(landmarks):
    if landmarks:
        # print("landmarks present")
        angles = get_elbow_angles(landmarks)

        if angles["left_elbow_angle"] > 160.0 and  angles["right_elbow_angle"] >  100 and landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y < landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y:
            return "arm_pointing"
        elif angles["left_elbow_angle"] > 160.0 and angles["right_elbow_angle"] < 15.0:
            return "arm_to_mouth"
        elif 65 < angles["left_elbow_angle"] < 90 and  65 < angles["right_elbow_angle"] < 90 and landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y < landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y:
            return "arms_over_head"
        elif 40 < angles["left_elbow_angle"] < 80 and  40 < angles["right_elbow_angle"] < 80 and landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y < landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y:
            return "arms_up"
        elif 35 < angles["left_elbow_angle"] < 60 and  35 < angles["right_elbow_angle"] < 60 and landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y > landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y:
            return "business_hands"
        elif 0.0 < angles["left_elbow_angle"] < 30 and  0.0 < angles["right_elbow_angle"] < 30 and landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y > landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y:
            return "choking"
        elif 30 < angles["left_elbow_angle"] < 50 and angles["right_elbow_angle"] < 10:
            return "pause"
        elif angles["left_elbow_angle"] > 160.0 and  20.0 < angles["right_elbow_angle"] < 30.0 and landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y > landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y:
            return "right_arm_up"
        elif landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y > landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST.value].y:
            return "left_arm_up"
        return "unknown_pose"
    return "error"

def get_mouth_state(landmarks):
    if landmarks.face_landmarks:
        upper_lip = landmarks.face_landmarks.landmark[13]
        lower_lip = landmarks.face_landmarks.landmark[14]
        # print(upper_lip.y)
        # print(lower_lip.y)
        if lower_lip.y - upper_lip.y >= 0.02:
            return "mouth_open"
        return "mouth_closed"
    return "error"

def testing():
    # photo1_landmarks = get_landmark_from_image("photo3.jpg")
    # photo1_angle = get_elbow_angles(get_landmark_from_image("photo3.jpg"))
    # print(photo1_angle)
    # photo1_hand_landmarks = get_hand_landmarks_from_image("thumbs up.jpg")
    # print("photo1 right hand landmarks")
    # print(photo1_hand_landmarks)
    # print(get_hand_state(photo1_hand_landmarks))
    # print(photo1_hand_landmarks[mp_holistic.HandLandmark.THUMB_TIP])
    test_photo = get_landmark_from_image("photos/choking.jpg")
    # print(get_elbow_angles(test_photo))
    # print(get_arms_state(test_photo))
    print(get_mouth_state(test_photo))
# testing()





