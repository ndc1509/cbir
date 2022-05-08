import json
import scipy.spatial.distance
from feature_extractor import extract_query_feature


# def sort_key(item):
#     return item[1]

def euclidean_dist(image, vector_list):
    query_fv = extract_query_feature(image)
    dissimilarity_dict = {}
    sorted_dict = {}
    results = []
    count = 0
    for data_obj in vector_list:
        vector = json.loads(data_obj['vector'])
        filename = data_obj['filename']
        dissimilarity_dict[filename] = scipy.spatial.distance.euclidean(query_fv, vector)
    for k, v in sorted(dissimilarity_dict.items(), key=lambda item: item[1]):
        sorted_dict[k] = v
    print(sorted_dict)
    for item in sorted_dict.items():
        if count == 5: break
        results.append(item)
        count += 1
    return results


# def manhattan_dist(image, vector_dict):
#     query_fv = extract_query_feature(image)
#     dissimilarity_dict = {}
#     for i in vector_dict:
#         dissimilarity_dict[i] = scipy.spatial.distance.cityblock(query_fv, vector_dict[i])
#     return dissimilarity_dict

