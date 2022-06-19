import cv2
import numpy
from matplotlib import pyplot
from skimage.feature import local_binary_pattern
from lib import get_image_dict


# trích rút đặc trưng lbp
# nhận được ảnh xám thể hiện kết cấu ảnh
# return lược đồ xám của ảnh kết cấu (là 1 mảng 256 phần tử)
def get_LBP(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp_image = local_binary_pattern(image, 8, 1, method='default')
    hist, _ = numpy.histogram(lbp_image.ravel(), density=True,
                              bins=256, range=(0, 256))
    # pyplot.plot(hist)
    # pyplot.show()
    return hist

# test
# hist = numpy.array([])
# image_dict = get_image_dict()
# for i in image_dict:
#     hist = get_LBP(image_dict[i])
