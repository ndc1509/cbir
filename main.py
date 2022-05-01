import cv2
import numpy
import feature_extractor
import redisDB
from preprocess import resize_image
# from similarity_measure import euclidean_dist, manhattan_dist

# MAIN

# Đọc ảnh đầu vào và trích rút đặc trưng
from similarity_measure import euclidean_dist, manhattan_dist

img = resize_image(cv2.imread('images/1.jpg'))
vector = feature_extractor.extract_query_feature(img)

# Cách 1


# Tìm kiếm
# query_vector = vector.astype(numpy.float32).tobytes()
# result = redisDB.query(query_vector)
# for doc in result.docs:
#     print(doc)

# Tạo index, trích xuất đặc trưng và lưu vào DB
# redisDB.create_flat_index()
# vector_dict = feature_extractor.extract_database_feature()
# redisDB.save_vectors(vector_dict)

# Xóa index
# redisDB.drop_index()

# ***

# Cách 2

# Lưu đặc trưng
vector_dict = feature_extractor.extract_database_feature()
redisDB.save_vectors_JSON(vector_dict)
# result1 = euclidean_dist(img, vector_dict=feature_extractor.extract_database_feature())
# print(result1)
# result2 = manhattan_dist(img, vector_dict=feature_extractor.extract_database_feature())
# print(result2)
