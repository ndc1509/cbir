import scipy.spatial.distance

from feature_extractor import extract_query_feature


def euclidean_dist(image, feature_vectors):
    query_fv = extract_query_feature(image)
    dissimilarity_arr = []
    for fv in feature_vectors:
        dissimilarity = scipy.spatial.distance.euclidean(query_fv, fv)
        dissimilarity_arr.append(dissimilarity)
    return dissimilarity_arr


def manhattan_dist(image, feature_vectors):
    query_fv = extract_query_feature(image)
    dissimilarity_arr = []
    for fv in feature_vectors:
        dissimilarity = scipy.spatial.distance.cityblock(query_fv, fv)
        dissimilarity_arr.append(dissimilarity)
    return dissimilarity_arr
