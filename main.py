import cv2
import numpy

import feature_extractor
from preprocess import get_image_list, resize_image
from rgb_histogram import get_rgb_histogram

#
# def get_GLCM(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     graycom = skimage.feature.graycomatrix(gray_image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)
#
#     # GLCM props
#     contrast = skimage.feature.graycoprops(graycom, 'contrast')
#     dissimilarity = skimage.feature.graycoprops(graycom, 'dissimilarity')
#     homogeneity = skimage.feature.graycoprops(graycom, 'homogeneity')
#     energy = skimage.feature.graycoprops(graycom, 'energy')
#     correlation = skimage.feature.graycoprops(graycom, 'correlation')
#     ASM = skimage.feature.graycoprops(graycom, 'ASM')
#
#     print("Contrast: {}".format(contrast))
#     print("Dissimilarity: {}".format(dissimilarity))
#     print("Homogeneity: {}".format(homogeneity))
#     print("Energy: {}".format(energy))
#     print("Correlation: {}".format(correlation))
#     print("ASM: {}".format(ASM))
#
#     for i in range(0, 4):
#         pyplot.plot(graycom[:, :, 0, i])
#         pyplot.show()
#
#     return graycom

# def get_contours(img):
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, thresh = cv2.threshold(gray_img, 127, 255, 0)
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
#     pyplot.imshow(img)
#     pyplot.show()
#     return contours


# MAIN
from similarity_measure import euclidean_dist, manhattan_dist

img = resize_image(cv2.imread('images/1.jpg'))
result1 = euclidean_dist(img, feature_vectors=feature_extractor.extract_database_feature())
result2 = manhattan_dist(img, feature_vectors=feature_extractor.extract_database_feature())
print(result1 + result2)