import cv2
import numpy
from matplotlib import pyplot
from preprocess import get_image_dict


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

# su dung lib numpy
# def get_rgb_histogram2(image):
#     colors = ('b', 'g', 'r')
#     hist_arr = numpy.array([])
#     channel_ids = (0, 1, 2)
#     for channel_id, c in zip(channel_ids, colors):
#         histogram, bin_edges = numpy.histogram(
#             image[:, :, channel_id], bins=256, range=(0, 256), density=True
#         )
#         hist_arr = numpy.concatenate((hist_arr, histogram))
#         pyplot.plot(bin_edges[0:-1], histogram, color=c)
#         # pyplot.plot(hist, color=color)
#         # print(hist.sum())
#     pyplot.show()
#     return hist_arr


# test
# hist = []
# hist2 = []
# image_dict = get_image_dict()
# for i in image_dict:
#     hist.append(get_rgb_histogram(image_dict[i]))
#     hist2.append(get_rgb_histogram2(image_dict[i]))
