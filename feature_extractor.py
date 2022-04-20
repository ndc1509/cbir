from feature_vector import combine_features
from hog import get_hog
from lbp import get_LBP
from preprocess import get_image_list
from rgb_histogram import get_rgb_histogram


def extract_query_feature(img):
    rgb_hist = get_rgb_histogram(img)
    lbp_hist = get_LBP(img)
    hog_hist = get_hog(img)
    fv = combine_features(rgb_hist, lbp_hist, hog_hist)
    print(fv.size)
    return fv


def extract_database_feature():
    vector = []
    for img in get_image_list():
        rgb_hist = get_rgb_histogram(img)
        lbp_hist = get_LBP(img)
        hog_hist = get_hog(img)
        vector.append(combine_features(rgb_hist, lbp_hist, hog_hist))
    return vector