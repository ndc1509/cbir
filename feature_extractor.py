from feature_vector import combine_features
from hog import get_hog
from lbp import get_LBP
from preprocess import get_image_dict
from rgb_histogram import get_rgb_histogram


# Rút đặc trưng ảnh query
# input img: ảnh đầu vào
# return vector đặc trưng
def extract_query_feature(img):
    rgb_hist = get_rgb_histogram(img)
    lbp_hist = get_LBP(img)
    hog_hist = get_hog(img)
    fv = combine_features(rgb_hist, lbp_hist, hog_hist)
    # print(fv.size)
    return fv


# Trích rút đặc trưng từ bộ dữ liệu ảnh
# Input image_dict: từ điển chứa tên ảnh - ảnh
# return từ điển chứa tên ảnh - vector đặc trưng tương ứng
def extract_database_feature():
    images = []
    vectors = []
    image_dict = get_image_dict()
    for i in image_dict:
        rgb_hist = get_rgb_histogram(image_dict[i])
        lbp_hist = get_LBP(image_dict[i])
        hog_hist = get_hog(image_dict[i])
        images.append(i)
        vectors.append(combine_features(rgb_hist, lbp_hist, hog_hist))
    vector_dict = dict(zip(images, vectors))
    return vector_dict
