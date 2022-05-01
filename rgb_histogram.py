import cv2
import numpy
from matplotlib import pyplot


def get_rgb_histogram(image):
    colors = ('b', 'g', 'r')
    hist_arr = numpy.array([])
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256]).ravel()
        # hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L2)
        hist /= hist.sum()
        hist_arr = numpy.concatenate((hist_arr, hist))
        # pyplot.plot(hist, color=color)
        # print(hist.sum())
    # pyplot.show()
    return hist_arr


# test
# hist = numpy.array([])
# image_dict = get_image_dict()
# for i in image_dict:
#     hist = get_rgb_histogram(image_dict[i])
