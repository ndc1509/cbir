import numpy


# Kết hợp các vector thành 1 vector 1d array 9124 phần tử
def combine_features(*features):
    feature_vector = numpy.array([])
    for feature in features:
        feature_vector = numpy.concatenate((feature_vector, feature))
    return feature_vector
