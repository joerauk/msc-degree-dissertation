import numpy as np
import cv2 as cv
import glob


hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

def hoggy(frame):
    boxes, weights = hog.detectMultiScale(frame, winStride=(4, 4), scale=1.0)
    return boxes, weights

def img():
    out = cv.VideoWriter(
        'hogoutput.avi',
        cv.VideoWriter_fourcc(*'DIVX'),
        20.,
        (640, 480))
    images = [cv.imread(f"output/frame{x}.jpg") for x in range(len(glob.glob1("output/","*.jpg")))]

    for frame in images:
        boxes, weights = hoggy(frame)
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        for (xA, yA, xB, yB) in boxes:
            cv.rectangle(frame, (xA, yA), (xB, yB),
                            (0, 255, 0), 2)
        out.write(frame.astype('uint8'))
        cv.imshow('frame', frame)
    out.release()
    cv.destroyAllWindows()
    cv.waitKey(1)