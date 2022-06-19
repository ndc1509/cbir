import numpy
from lbp import get_LBP
from lib import get_image_dict
from rgb_histogram import get_rgb_histogram


# Kết hợp các vector thành 1 vector 1024 chiều
def combine_features(*features):
    feature_vector = numpy.array([])
    for feature in features:
        feature_vector = numpy.concatenate((feature_vector, feature))
    return feature_vector


# Rút đặc trưng ảnh query
# input img: ảnh query
# return vector đặc trưng
def extract_query_feature(img):
    rgb_hist = get_rgb_histogram(img)
    lbp_hist = get_LBP(img)
    fv = combine_features(rgb_hist, lbp_hist)
    # print(fv.size)
    return fv


# Trích rút đặc trưng từ bộ dữ liệu ảnh
# return từ điển chứa tên ảnh - vector đặc trưng tương ứng
def extract_database_feature():
    images = []
    vectors = []
    image_dict = get_image_dict()
    for i in image_dict:
        rgb_hist = get_rgb_histogram(image_dict[i])
        lbp_hist = get_LBP(image_dict[i])
        images.append(i)
        vectors.append(combine_features(rgb_hist, lbp_hist))
    vector_dict = dict(zip(images, vectors))
    return vector_dict
