import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt

# code used is directly taken from official OpenCV documentation - not developed further due to time constraints
# this module exists only to demonstrate original intention for project, and I am not claiming this work as my own


def img():
    # how to access each image in the folder, and de-noise them - then can detect features etc
    images = [cv.imread(f"output/frame{x}.jpg") for x in range(len(glob.glob1("output/","*.jpg")))]

    # convert all to float64
    f_64 = [np.float64(i) for i in images]

    # create a noise of variance 25
    noise = np.random.randn(*f_64[1].shape)*10

    # Add this noise to images
    noisy = [i+noise for i in f_64]

    # Convert back to uint8
    noisy = [np.uint8(np.clip(i,0,255)) for i in noisy]


    # Denoise 3rd frame considering all the 5 frames
    # for i in images:
    #     dst = cv.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)
    #     plt.subplot(131),plt.imshow(images[2],'gray')
    #     plt.subplot(132),plt.imshow(noisy[2],'gray')
    #     plt.subplot(133),plt.imshow(dst,'gray')
    #     plt.show()
    #     # break