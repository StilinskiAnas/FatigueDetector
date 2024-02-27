from scipy.spatial import distance
from imutils import face_utils, resize
from dlib import get_frontal_face_detector, shape_predictor
import cv2


class FaceAction:
    tot = 0
    detect = get_frontal_face_detector()
    # predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor_path = "C:/Users/Stilinski/Code/FatigueDetector/FatigueDetector/shape_predictor_68_face_landmarks.dat"
    predict = shape_predictor(predictor_path)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth):
        A = distance.euclidean(mouth[13], mouth[19])
        B = distance.euclidean(mouth[14], mouth[18])
        C = distance.euclidean(mouth[15], mouth[17])
        D = distance.euclidean(mouth[12], mouth[16])
        mar = (A + B + C) / (2.0 * D)
        return mar
    def run_frame(self, frame):
        results = []
        resized_frame = resize(frame, width=450)
        gray = resized_frame
        subjects = self.detect(gray, 0)

        if len(subjects) == 0:
            return results  # No faces detected, return an empty list

        for subject in subjects:
            shape = self.predict(gray, subject)
            shape = face_utils.shape_to_np(shape)

            # Scale coordinates based on the resizing factor
            scale_factor_x = frame.shape[1] / resized_frame.shape[1]
            scale_factor_y = frame.shape[0] / resized_frame.shape[0]
            leftEye = (shape[self.lStart:self.lEnd] * [scale_factor_x, scale_factor_y]).astype(int)
            rightEye = (shape[self.rStart:self.rEnd] * [scale_factor_x, scale_factor_y]).astype(int)
            mouth = (shape[self.mStart:self.mEnd] * [scale_factor_x, scale_factor_y]).astype(int)

            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = self.mouth_aspect_ratio(mouth)

            # print(f'leftEye: {shape[self.lStart:self.lEnd]}\n rightEye: {shape[self.rStart:self.rEnd]}\n mouth: {shape[self.mStart:self.mEnd]}')

            # Append a tuple containing (ear, mar) for each person
            results.append((ear, mar, leftEye, rightEye, mouth, leftEAR, rightEAR))

        return results
