import numpy as np
import cv2

#etiqueta_machine = cv2.CascadeClassifier('etiqueta_machinepart.xml')
etiqueta_pre = cv2.CascadeClassifier('etiqueta_precaucion.xml')
etiqueta_ne = cv2.CascadeClassifier('etiqueta_negra.xml')
#etiqueta_at = cv2.CascadeClassifier('fru.xml')
img = cv2.imread('warning2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
etiquetapre = etiqueta_pre.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=18, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
#etiquetamach = etiqueta_machine.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=7, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
etiquetane = etiqueta_ne.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=15, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
#etiquetaatras = etiqueta_at.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=2, minSize=(2, 2), flags=cv2.CASCADE_SCALE_IMAGE)
for (x, y, w, h) in etiquetapre:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
# (x, y, w, h) in etiquetamach:
#    cv2.rectangle(img, (x, y), (x+w, y+h), (140, 0, 0), 2)
for (x, y, w, h) in etiquetane:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#for (x, y, w, h) in etiquetaatras:
#    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 200, 180), 2)

cv2.imshow('img', img)
cv2.imwrite('img.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
