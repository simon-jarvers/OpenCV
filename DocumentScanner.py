import cv2
import numpy as np
import urllib.request

# frameWidth, frameHeight = 1440, 1080
imgWidth, imgHeiht = 960, 720

URL = "http://192.168.0.5:8080/shot.jpg"

# cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# cap.set(10, 150)


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def preprocessing(img):
    kernel = np.ones(5)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 100, 200)
    imgDilation = cv2.dilate(imgCanny, kernel=kernel, iterations=2)
    imgErode = cv2.erode(imgDilation, kernel=kernel, iterations=1)
    return imgErode

def getContour(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            if area < 100000:
                cv2.drawContours(imgContour, cnt, -1, (0, 0, 255), 3)
            elif 100000 < area < 200000:
                cv2.drawContours(imgContour, cnt, -1, (0, 255, 255), 3)
            elif area > 200000:
                cv2.drawContours(imgContour, cnt, -1, (0, 255, 0), 3)

            print(area)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area

    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest


def reorder(points):
    points_out = np.zeros((4, 1, 2), np.int32)
    points = points.reshape((4, 2))
    add = points.sum(axis=1)

    points_out[0] = points[np.argmin(add)]
    points_out[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    points_out[2] = points[np.argmin(diff)]
    points_out[1] = points[np.argmax(diff)]

    return points_out


def getWarp(img, biggest):
    if not biggest.size:
        return img
    biggest = reorder(biggest)
    p1 = np.float32(biggest)
    p2 = np.float32([[0, 0], [0, imgHeiht], [imgWidth, 0], [imgWidth, imgHeiht]])
    matrix = cv2.getPerspectiveTransform(p1, p2)
    imgOutput = cv2.warpPerspective(img, matrix, (imgWidth, imgHeiht))
    return imgOutput


while True:
    img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = cv2.resize(img, (imgWidth, imgHeiht))
    imgContour = img.copy()

    imgThresh = preprocessing(img)
    biggest = getContour(imgThresh)
    if biggest.size:
        imgWarp = getWarp(img, biggest)
        image_array = ([img, imgThresh],
                       [imgContour, imgWarp])
        cv2.imshow("Scanned Document", imgWarp)
    else:
        image_array = ([img, imgThresh],
                       [imgContour, img])

    stacked_images = stackImages(0.5, image_array)

    cv2.imshow("Workflow", stacked_images)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
