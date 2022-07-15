import numpy as np
import cv2 as cv
import glob

class Hog():

    def __init__(self):
        self.hoggy = cv.HOGDescriptor()
        self.hoggy.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    def getWeightedBoxes(self, frame):
        boxes, weights = self.hoggy.detectMultiScale(frame, winStride=(4, 4), scale=1.0)
        return boxes, weights
        
    def drawBoxes(self, boxes, frame):        
        for (xA, yA, xB, yB) in boxes:
            cv.rectangle(frame, (xA, yA), (xB, yB),
                            (0, 255, 0), 2)
        return frame

    def process(self, frame):
        frame = frame.astype('uint8')
        boxes, _ = self.getWeightedBoxes(frame)
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        
        frame = self.drawBoxes(boxes, frame)
        return frame