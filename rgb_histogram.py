import cv2
import numpy
import skimage.io
from matplotlib import pyplot
from lib import get_image_dict


# su dung lib cv2
# def get_rgb_histogram(image):
#     colors = ('r', 'g', 'b')
#     hist_arr = numpy.array([])
#     for i, color in enumerate(colors):
#         hist = cv2.calcHist([image], [i], None, [256], [0, 256]).ravel()
#         # hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L2)
#         hist /= hist.sum()
#         hist_arr = numpy.concatenate((hist_arr, hist))
#         # pyplot.plot(hist, color=color)
#         # print(hist.sum())
#     # pyplot.show()
#     return hist_arr

# su dung lib numpy
# return vector lược đồ màu rgb của ảnh (256 x 3 = 768 phần tử)
def get_rgb_histogram(image):
    colors = ('r', 'g', 'b')
    hist_arr = numpy.array([])
    channel_ids = (0, 1, 2)
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = numpy.histogram(
            image[:, :, channel_id], bins=256, range=(0, 256), density=True
        )
        hist_arr = numpy.concatenate((hist_arr, histogram))
        # pyplot.plot(bin_edges[0:-1], histogram, color=c)
        # pyplot.plot(hist, color=color)
        # print(hist.sum())
    # pyplot.show()
    return hist_arr

# # test
# hist = []
# hist2 = []
# image_dict = get_image_dict()
# # for i in image_dict:
# img1 = cv2.imread('query_images/dongvangchieu (3).jpg')
# img2 = skimage.io.imread('query_images/dongvangchieu (3).jpg')
# hist.append(get_rgb_histogram(img2))
# hist2.append(get_rgb_histogram2(img2))
