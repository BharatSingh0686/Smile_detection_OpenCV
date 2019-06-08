# -*- coding: utf-8 -*-
import cv2

smile_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')


def detectsmile(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            print("Smile : ", smiles)
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 3)
    return frame


if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)
    while True:
        _, frame = video_capture.read()
        grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = detectsmile(grayimg, frame)
        cv2.imshow("Face Detection ", canvas)

        # The control breaks once q key is pressed
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    # Release the capture once all the processing is done.
    video_capture.release()
    cv2.destroyAllWindows()
