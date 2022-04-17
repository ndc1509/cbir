import os
import cv2
import numpy as np
import skimage.feature
from matplotlib import pyplot


def resize_image(image):
    return cv2.resize(image, (256, 256))


def get_image_list():
    image_list = []
    image_name_list = os.listdir("images")
    for image_name in image_name_list:
        i = cv2.imread("images/" + image_name)
        image_list.append(resize_image(i))
        # cv2.imshow(image_name, i)
    return image_list


def get_rgb_histogram(image):
    hist_full = cv2.calcHist([image], [0], None, [256], [0, 256])
    pyplot.plot(hist_full)
    pyplot.show()
    return hist_full


# Contrast: Measures the local variations in the gray-level co-occurrence matrix.
# Correlation: Measures the joint probability occurrence of the specified pixel pairs.
# Energy: Provides the sum of squared elements in the GLCM. Also known as uniformity or the angular second moment.
# Homogeneity: Measures the closeness of the distribution of elements in the GLCM to the GLCM diagonal.


def get_GLCM(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    graycom = skimage.feature.graycomatrix(gray_image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)

    # GLCM props
    contrast = skimage.feature.graycoprops(graycom, 'contrast')
    dissimilarity = skimage.feature.graycoprops(graycom, 'dissimilarity')
    homogeneity = skimage.feature.graycoprops(graycom, 'homogeneity')
    energy = skimage.feature.graycoprops(graycom, 'energy')
    correlation = skimage.feature.graycoprops(graycom, 'correlation')
    ASM = skimage.feature.graycoprops(graycom, 'ASM')

    print("Contrast: {}".format(contrast))
    print("Dissimilarity: {}".format(dissimilarity))
    print("Homogeneity: {}".format(homogeneity))
    print("Energy: {}".format(energy))
    print("Correlation: {}".format(correlation))
    print("ASM: {}".format(ASM))

    for i in range(0,4):
        pyplot.plot(graycom[:,:,0,i])
        pyplot.show()

    return graycom


def get_contours(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    pyplot.imshow(img)
    pyplot.show()

# MAIN
list = get_image_list()
for img in list:
    #get_rgb_histogram(img)
    #get_GLCM(img)
    get_contours(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
