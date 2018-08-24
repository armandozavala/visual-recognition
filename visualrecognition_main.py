import numpy as np
import cv2

etiqueta_machine = cv2.CascadeClassifier('etiqueta_machinepart.xml')
etiqueta_pre = cv2.CascadeClassifier('etiqueta_precaucion.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    etiquetapre = etiqueta_pre.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=3, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    etiquetamach = etiqueta_pre.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=3, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in etiquetapre:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
    for (x, y, w, h) in etiquetamach:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 200, 0), 2)
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
