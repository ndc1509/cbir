import numpy
from matplotlib import pyplot
from skimage.feature import hog

from preprocess import get_image_list


def get_hog(img):
    fd, hog_img = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                      visualize=True, feature_vector=True, channel_axis=-1)
    pyplot.imshow(hog_img, cmap="gray")
    pyplot.show()
    print(fd.sum())
    return fd


# test
hist = numpy.array([])
for img in get_image_list():
    hist = get_hog(img)