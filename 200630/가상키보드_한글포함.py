from shapely.geometry import Point
import cv2 as cv
import numpy as np
from shapely.geometry.polygon import Polygon
from PIL import ImageFont, ImageDraw, Image


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []

    rects[:, 2:] += rects[:, :2]

    return rects


def removeFaceAra(img, cascade):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    rects = detect(gray, cascade)

    height, width = img.shape[:2]

    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1 - 10, 0), (x2 + 10, height), (0, 0, 0), -1)

    return img


def make_mask_image(img_bgr):
    img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

    # img_h,img_s,img_v = cv.split(img_hsv)

    low = (0, 30, 0)
    high = (15, 255, 255)

    img_mask = cv.inRange(img_hsv, low, high)

    return img_mask


def distanceBetweenTwoPoints(start, end):
    x1, y1 = start
    x2, y2 = end

    return int(np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))


def calculateAngle(A, B):
    A_norm = np.linalg.norm(A)
    B_norm = np.linalg.norm(B)
    C = np.dot(A, B)

    angle = np.arccos(C / (A_norm * B_norm)) * 180 / np.pi

    return angle


def findMaxArea(contours):
    max_contour = None
    max_area = -1

    for contour in contours:
        area = cv.contourArea(contour)

        x, y, w, h = cv.boundingRect(contour)

        if (w * h) * 0.4 > area:
            continue

        if w > h:
            continue

        if area > max_area:
            max_area = area
            max_contour = contour

    if max_area < 10000:
        max_area = -1

    return max_area, max_contour


def getFingerPosition(max_contour, img_result, debug):
    points1 = []

    # STEP 6-1
    M = cv.moments(max_contour)

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    max_contour = cv.approxPolyDP(max_contour, 0.02 * cv.arcLength(max_contour, True), True)
    hull = cv.convexHull(max_contour)

    for point in hull:
        if cy > point[0][1]:
            points1.append(tuple(point[0]))

    if debug:
        cv.drawContours(img_result, [hull], 0, (0, 255, 0), 2)
        for point in points1:
            cv.circle(img_result, tuple(point), 15, [0, 0, 0], -1)

    # STEP 6-2
    hull = cv.convexHull(max_contour, returnPoints=False)
    defects = cv.convexityDefects(max_contour, hull)

    if defects is None:
        return -1, None

    points2 = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])

        angle = calculateAngle(np.array(start) - np.array(far), np.array(end) - np.array(far))

        if angle < 90:
            if start[1] < cy:
                points2.append(start)

            if end[1] < cy:
                points2.append(end)

    if debug:
        cv.drawContours(img_result, [max_contour], 0, (255, 0, 255), 2)
        for point in points2:
            cv.circle(img_result, tuple(point), 20, [0, 255, 0], 5)

    # STEP 6-3
    points = points1 + points2
    points = list(set(points))

    # STEP 6-4
    new_points = []
    for p0 in points:

        i = -1
        for index, c0 in enumerate(max_contour):
            c0 = tuple(c0[0])

            if p0 == c0 or distanceBetweenTwoPoints(p0, c0) < 20:
                i = index
                break

        if i >= 0:
            pre = i - 1
            if pre < 0:
                pre = max_contour[len(max_contour) - 1][0]
            else:
                pre = max_contour[i - 1][0]

            next = i + 1
            if next > len(max_contour) - 1:
                next = max_contour[0][0]
            else:
                next = max_contour[i + 1][0]

            if isinstance(pre, np.ndarray):
                pre = tuple(pre.tolist())
            if isinstance(next, np.ndarray):
                next = tuple(next.tolist())

            angle = calculateAngle(np.array(pre) - np.array(p0), np.array(next) - np.array(p0))

            if angle < 90:
                new_points.append(p0)

    return 1, new_points

def listToString(s):
    str1 = " "
    return (str1.join(s))

Flag = [0, 0, 0, 0, 0]
f = open("letter.txt", "w+")

x1 = 40
y1 = 50
x2 = 600
y2 = 400
width = x2 - x1
height = y2 - y1
term = 50

# Fs = ["MOM", "DAD", "LOVE", "YOU"]
Fs = ["엄마", "아빠", "LOVE", "YOU"]

txt = []

def process(img_bgr, debug):
    img_result = img_bgr.copy()
    img = np.zeros((200,400,3),np.uint8)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    k11 = (width // 5 * 1 - term + x1, height // 2 - term + y1)
    k12 = (width // 5 * 1 + term + x1, height // 2 + term + y1)
    k21 = (width // 5 * 2 - term + x1, height // 2 - term + y1)
    k22 = (width // 5 * 2 + term + x1, height // 2 + term + y1)
    k31 = (width // 5 * 3 - term + x1, height // 2 - term + y1)
    k32 = (width // 5 * 3 + term + x1, height // 2 + term + y1)
    k41 = (width // 5 * 4 - term + x1, height // 2 - term + y1)
    k42 = (width // 5 * 4 + term + x1, height // 2 + term + y1)

    key1 = Polygon([(k11[0], k11[1]), (k12[0], k11[1]), (k12[0], k12[1]), (k11[0], k12[1])])
    key2 = Polygon([(k21[0], k21[1]), (k22[0], k21[1]), (k22[0], k22[1]), (k21[0], k22[1])])
    key3 = Polygon([(k31[0], k31[1]), (k32[0], k31[1]), (k32[0], k32[1]), (k31[0], k32[1])])
    key4 = Polygon([(k41[0], k41[1]), (k42[0], k41[1]), (k42[0], k42[1]), (k41[0], k42[1])])

    # STEP 1
    img_bgr = removeFaceAra(img_bgr, cascade)

    # STEP 2
    img_binary = make_mask_image(img_bgr)

    # STEP 3: all the pixels near boundary will be discarded depending upon the size of kernel
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))  # Elliptical Kernel
    # MORPH_CLOSE: Dilation followed by Erosion; useful in closing small holes inside the foreground objects, or small black points on the object.
    img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, 1)
    #cv.imshow("Binary", img_binary)

    # STEP 4
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)  # finds contours in a binary image where each contour is stored as a vector of points

    if debug:
        for cnt in contours:
            cv.drawContours(img_result, [cnt], 0, (255, 0, 0), 3)

            # STEP 5
    max_area, max_contour = findMaxArea(contours)

    if max_area == -1:
        return img_result

    if debug:
        cv.drawContours(img_result, [max_contour], 0, (0, 0, 255), 3)

        # STEP 6
    ret, points = getFingerPosition(max_contour, img_result, debug)

    if points != None:
        for point in points:
            if key1.contains(Point(point)):
                # print(F1)
                Flag[0] += 1
                if Flag[0] % 20 == 19:
                    print(Fs[0])
                    txt.append(Fs[0])
            elif key2.contains(Point(point)):
                Flag[0] = 0
                Flag[1] += 1
                Flag[2] = 0
                Flag[3] = 0
                if Flag[1] % 20 == 19:
                    print(Fs[1])
                    txt.append(Fs[1])
            elif key3.contains(Point(point)):
                Flag[0] = 0
                Flag[1] = 0
                Flag[2] += 1
                Flag[3] = 0
                if Flag[2] % 20 == 19:
                    print(Fs[2])
                    txt.append(Fs[2])
            elif key4.contains(Point(point)):
                Flag[0] = 0
                Flag[1] = 0
                Flag[2] = 0
                Flag[3] += 1
                if Flag[3] % 20 == 19:
                    print(Fs[3])
                    txt.append(Fs[3])
            else:
                Flag[4] += 1
                if Flag[4] % 20 == 19:
                    print("!")
                    f.write("!")

    # STEP 7
    if ret > 0 and len(points) > 0:
        for point in points:
            cv.circle(img_result, point, 10, [255, 0, 255], 5)

    return img_result

cascade = cv.CascadeClassifier(cv.samples.findFile("haarcascade_frontalface_alt.xml"))

cap = cv.VideoCapture(0)

fontpath = "C:/Users/KOSTA/Desktop/NGULIM.TTF"
font = ImageFont.truetype(fontpath, 30)

while True:

    ret, img_bgr = cap.read()

    if ret == False:
        break

    img_result = process(img_bgr, debug=False)

    key = cv.waitKey(1)
    if key == 27:
        break

    inner_x1 = []
    inner_y1 = []
    inner_x2 = []
    inner_y2 = []

    for i in range(4):
        inner_x1.append(width // 5 * (i + 1) - term + x1)
        inner_y1.append(height // 2 + term + y1)
        inner_x2.append(width // 5 * (i + 1) + term + x1)
        inner_y2.append(height // 2 - term + y1)


    cv.rectangle(img_result, (x1, y1), (x2, y2), (255, 0, 0), 4)

    img_pil = Image.fromarray(img_result)
    draw = ImageDraw.Draw(img_pil)

    for i in range(4):
        draw.text((inner_x1[i]+20, inner_y1[i]-45), Fs[i], font=font, fill=(255, 0, 0, 0))

    img_result = np.array(img_pil)

    for i in range(4):
        cv.rectangle(img_result, (inner_x1[i], inner_y1[i]), (inner_x2[i], inner_y2[i]), (255, 0, 0), 4)

    img = np.zeros((400, 800, 3), np.uint8)
    img_txt = Image.fromarray(img)
    draw = ImageDraw.Draw(img_txt)

    str = listToString(txt)
    draw.text((10, 20), str, font=font, fill=(255, 255, 255, 0))

    img_txt = np.array(img_txt)

    cv.imshow("Result", img_result)
    cv.imshow("txt", img_txt)

f.close()
cap.release()
cv.destroyAllWindows()
