import cv2

import feature_extractor
import redisDB
from preprocess import resize_image
from similarity_measure import euclidean_dist

# MAIN

# Đọc ảnh đầu vào và
img = resize_image(cv2.imread('images/2.jpg'))

# # Trích rút đặc trưng
# vector_dict = feature_extractor.extract_database_feature()
# # Lưu đặc trưng
# redisDB.save_vectors_JSON(vector_dict)

# Tìm kiếm
vectors = redisDB.get_vectors()
results = euclidean_dist(img, vectors)
print("Top 10 ảnh giống nhất: ")
for result in results:
    print(result)
