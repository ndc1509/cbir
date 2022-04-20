import os

import cv2


def resize_image(image):
    return cv2.resize(image, (256, 256))


def get_image_list():
    image_list = []
    image_name_list = os.listdir("images")
    for image_name in image_name_list:
        i = cv2.imread("images/" + image_name)
        image_list.append(cv2.resize(i, (256, 256)))
        # cv2.imshow(image_name, i)
    return image_list