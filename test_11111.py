import cv2
import numpy as np

img = cv2.imread("dataset/PIL/train/image/180109_01_000293.jpg")
 image = cv2.resize(img, (512, 512))
cv2.imshow("dataset/PIL/train/image/180109_01_000293.jpg",img)

cv2.waitKey()
cv2.destroyAllWindows()
