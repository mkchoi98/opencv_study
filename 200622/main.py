import dlib
import cv2 as cv
import numpy as np
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
cap = cv.VideoCapture(0)

# range는 끝값이 포함안됨
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))

# index = ALL
index = LEFT_EYE + RIGHT_EYE

flag = 0
cnt = 0

start = time.time()

while True:
    ret, img_frame = cap.read()

    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)

    dets = detector(img_gray, 1)

    for face in dets:

        shape = predictor(img_frame, face)  # 얼굴에서 68개 점 찾기

        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)

        for i, pt in enumerate(list_points[index]):
            pt_pos = (pt[0], pt[1])
            cv.circle(img_frame, pt_pos, 2, (255, 0, 0), -1)

    #ear = np.abs(list_points[index][2]-list_points[index][6])+np.abs(list_points[index][3]-list_points[index][5]) / (2 * np.abs(list_points[1] - list_points[4]))
    ear = (np.abs(list_points[index][1] - list_points[index][5]) + np.abs(
        list_points[index][2] - list_points[index][4])) / np.abs(2.0 * (list_points[0] - list_points[3]))

    print(ear)

    if (ear[1] < 0.1):
        flag = 1
        for i, pt in enumerate(list_points[index]):
            pt_pos = (pt[0], pt[1])
            cv.circle(img_frame, pt_pos, 2, (0, 0, 255), -1)

    if flag == 1:
        if ear[1] > 0.1 :
            cnt += 1
            flag = 0

    cv.putText(img_frame, "Blinks : {:d} ".format(cnt), (0, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    cv.putText(img_frame, "/ {:.2f} sec".format((time.time() - start)), (170, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    cv.imshow('result', img_frame)

    key = cv.waitKey(1)

    if (key == 27):
        break


    '''elif key == ord('1'):
        index = ALL
    elif key == ord('2'):
        index = LEFT_EYEBROW + RIGHT_EYEBROW
    elif key == ord('3'):
        index = LEFT_EYE + RIGHT_EYE
    elif key == ord('4'):
        index = NOSE
    elif key == ord('5'):
        index = MOUTH_OUTLINE + MOUTH_INNER
    elif key == ord('6'):
        index = JAWLINE'''

print(cnt)

cap.release()