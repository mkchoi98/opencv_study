'''import numpy as np
import cv2
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def auto_scan_image_via_webcam():

    try : cap = cv2.VideoCapture(0)

    except : print('Cannot load Camera')

    while True :
        flag = 0

        ret, frame = cap.read()

        if not ret:
            print('Cannot load Camera')
            break

        k = cv2.waitKey(10)

        if k == 27 : break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(gray, 75, 200)

        print("edge detection")

        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            screenCnt = []

            if len(approx) == 4:
                contourSize = cv2.contourArea(approx)

                camSize = frame.shape[0] * frame.shape[1]
                ratio = contourSize / camSize

                print(contourSize)
                print(camSize)
                print(ratio)

                ret, frame = cap.read()

                if ratio > 0.1:
                    screenCnt = approx
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    img = Image.fromarray(gray)  # frame으로 넣어도 된다.
                    
                    txt = pytesseract.image_to_string(img, lang='kor+eng')
                    print(txt)

                    return 

        if len(screenCnt) == 0:
            cv2.imshow('WebCam', frame)

            continue

        else :
            cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 2)
            cv2.imshow('WebCam', frame)

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey()

if __name__ == '__main__' :
    auto_scan_image_via_webcam()
'''

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

image = Image.open('시은이명함2.jpg')
print(pytesseract.image_to_string(image, lang='kor+eng'))
