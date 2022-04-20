import numpy

from hog import get_hog
from lbp import get_LBP
from preprocess import get_image_list
from rgb_histogram import get_rgb_histogram


def combine_features(*features):
    feature_vector = numpy.array([])
    for feature in features:
        feature_vector = numpy.concatenate((feature_vector, feature))
    return feature_vector


# test
vector = []
for img in get_image_list():
    rgb_hist = get_rgb_histogram(img)
    lbp_hist = get_LBP(img)
    hog_hist = get_hog(img)
    vector.append(combine_features(rgb_hist, lbp_hist, hog_hist))

