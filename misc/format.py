import cv2 as cv
import os
from pathlib import Path


def img():
    cap = cv.VideoCapture('output.avi')
    if not os.path.exists('../output'):
        os.mkdir('../output')
    else:
        [f.unlink() for f in Path("../output/").glob("*") if f.is_file()]

    s = 0
    n = 0
    frameRate = .45

    while True:

        ret, frame = cap.read()
        if ret == False:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imwrite('output/frame'+str(n)+'.jpg', gray)

        n += 1
        s += frameRate

    cap.release()
    cv.destroyAllWindows()