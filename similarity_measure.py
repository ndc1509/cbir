import scipy.spatial.distance

from feature_extractor import extract_query_feature


def euclidean_dist(image, vector_dict):
    query_fv = extract_query_feature(image)
    dissimilarity_dict = {}
    for i in vector_dict:
        dissimilarity_dict[i] = scipy.spatial.distance.euclidean(query_fv, vector_dict[i])
    return dissimilarity_dict


def manhattan_dist(image, vector_dict):
    query_fv = extract_query_feature(image)
    dissimilarity_dict = {}
    for i in vector_dict:
        dissimilarity_dict[i] = scipy.spatial.distance.cityblock(query_fv, vector_dict[i])
    return dissimilarity_dict
