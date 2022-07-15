import cv2 as cv


def img():
    vid = cv.VideoCapture(0)

    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    out = cv.VideoWriter('output.avi', fourcc, 45, (640, 480))

    while True:
        ret, frame = vid.read()
        if not ret:
            print('error')
            break

        out.write(frame)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    vid.release()
    out.release()
    cv.destroyAllWindows()