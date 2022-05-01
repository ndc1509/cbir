import os
import cv2


# Resize ảnh
def resize_image(image):
    return cv2.resize(image, (256, 256))


# return từ điển chứa ảnh - tên ảnh
def get_image_dict():
    image_dict = {}
    image_name_list = os.listdir("images")
    for image_name in image_name_list:
        i = cv2.imread("images/" + image_name)
        image_dict[image_name] = cv2.resize(i, (256, 256))
        # cv2.imshow(image_name, i)
    return image_dict
