import cv2
import skimage
from matplotlib import pyplot
import feature_extractor
import redisDB
from lib import resize_image, get_image_dict, show_result
from similarity_measure import euclidean_dist

# MAIN

# Đọc ảnh đầu vào và resize ảnh
img = resize_image(skimage.io.imread('query_images/dongvangchieu (1).jpg'))

# # Trích rút đặc trưng CSDL ảnh
# vector_dict = feature_extractor.extract_database_feature()
# # Lưu đặc trưng vào redis
# redisDB.save_vectors_JSON(vector_dict)

# Tìm kiếm
vectors = redisDB.get_vectors()
results = euclidean_dist(img, vectors)
print("Top 10 ảnh giống nhất: ")
for result in results:
    show_result(result)
    print(result)
