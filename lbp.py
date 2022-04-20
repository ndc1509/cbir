import cv2
import numpy
from matplotlib import pyplot
from skimage.feature import local_binary_pattern

from preprocess import get_image_list


def get_LBP(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_image = local_binary_pattern(image, 8, 1, method='default')
    pyplot.imshow(lbp_image, cmap="gray")
    pyplot.show()

    hist, _ = numpy.histogram(lbp_image.ravel(), density=True,
                              bins=256, range=(0, 255))
    return hist


# test
hist = numpy.array([])
for img in get_image_list():
    hist = get_LBP(img)
