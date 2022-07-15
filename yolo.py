import cv2 as cv
import numpy as np
import time


class Yolo():
    def __init__(self, weightsFile, configFile):
        net = cv.dnn.readNet(weightsFile, configFile)
        classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layers = net.getLayerNames()
        outLayers = [layers[i[0]-1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        self.model = net
        self.classes = classes
        self.colors = colors
        self.output_layers = outLayers

    def forwardNet(self, blob):
        self.model.setInput(blob)
        out = self.model.forward(self.output_layers)
        return out
    
    
    def drawBoxesToFrame(self, boxes, confs, ids, frame):
        '''
        Draw boxes labelled with their definitions to the frame
        '''
        indexes = cv.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
        font = cv.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i not in indexes: continue
            x, y, w, h = boxes[i]
            label = str(self.classes[ids[i]])
            color = self.colors[i]

            cv.rectangle(frame, (x, y), (x + w , y + h), color, 2)
            cv.putText(frame, label, (x, y - 5), font, 1, color, 1)
        return frame

    def getBoxesFromOutput(self, outputs, width, height):
        boxes = []
        confs = []
        ids = []

        for output in outputs:
            for detection in output:
                # skip header
                scores = detection[5:]
                id = np.argmax(scores)
                confidence = scores[id]

                # 30% confidence
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confs.append(float(confidence))
                    ids.append(id)
        return boxes, confs, ids
    
    def process(self, frame):
        h, w, _ = frame.shape
        blob = cv.dnn.blobFromImage(frame, scalefactor=0.004, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        outputs = self.forwardNet(blob)
        boxes, confs, ids = self.getBoxesFromOutput(outputs, w, h)

        return self.drawBoxesToFrame(boxes, confs, ids, frame)



if __name__ == '__main__':
    pass