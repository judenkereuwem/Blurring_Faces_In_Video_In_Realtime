
# Title: Real-time face blur in video
# Author: Jude Nkereuwem
# Date: 27/10/22

import cv2

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)

while True:
    check, frame = video_capture.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = cascade.detectMultiScale(gray_image, scaleFactor=3.0, minNeighbors=4)

    for x, y, w, h in face:
        image = cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 1)
        image[y:y+h, x:x+w] = cv2.medianBlur(image[y:y+h, x:x+w], 35)

    cv2.imshow('Blurred Faces', frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
