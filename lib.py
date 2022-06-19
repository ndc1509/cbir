import os
import cv2
import skimage.io
from matplotlib import pyplot


# Resize ảnh
def resize_image(image):
    return cv2.resize(image, (256, 256))


# return từ điển chứa ảnh - tên ảnh
def get_image_dict():
    image_dict = {}
    image_name_list = os.listdir("images")
    for image_name in image_name_list:
        i = skimage.io.imread("images/" + image_name)
        image_dict[image_name] = i
        # cv2.imshow(image_name, i)
    return image_dict


# hiển thị ảnh kết quả bằng pyplot
def show_result(result):
    skimage.io.imshow("images/" + result[0])
    pyplot.title(result[0] + " " + str(result[1]), {'family':'Arial','color':'black','size':10})
    pyplot.show()