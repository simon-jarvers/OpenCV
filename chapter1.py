import cv2
import numpy as np

# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
# cap.set(10, 200)
#
# while True:
#     Success, img = cap.read()
#     cv2.imshow("Webcam", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

img = cv2.imread("venv\Resources\lena.png")
kernel = np.ones((5, 5), np.uint8)

imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imBlur = cv2.GaussianBlur(imGray, (7, 7), 0)
imCanny = cv2.Canny(img, 150, 200)
imDilation = cv2.dilate(imCanny, kernel=kernel, iterations=1)
imEroded = cv2.erode(imDilation, kernel=kernel, iterations=1)

#cv2.imshow("Gray Image", imGray)
#cv2.imshow("Blur Image", imBlur)
cv2.imshow("Canny Image", imCanny)
cv2.imshow("Dialtion Image", imDilation)
cv2.imshow("Erosion Image", imEroded )
cv2.waitKey(0)
