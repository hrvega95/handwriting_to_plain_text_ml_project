from imutils import contours
from skimage import measure
import sklearn
import PIL.Image
import sys
import cv2
import os.path
import numpy as np
import imutils
import time


def convert_images_to_bw(image):
    (thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw

def make_all_images_bw():
    root = os.path.dirname(__file__) + "/../Data/"
    for name_w_path in get_all_data_png(root):
        im_gray = cv2.imread(name_w_path, cv2.IMREAD_GRAYSCALE)
        im_bw = convert_images_to_bw(im_gray)
        cv2.imwrite(name_w_path, im_bw)

def crop_data_all_images(data_type):
    root = os.path.dirname(__file__) + "/../Data/" + str(data_type)
    for file_w_path in get_all_data_png(root):
        img = cv2.imread(file_w_path,cv2.IMREAD_GRAYSCALE)
        rect = crop_single_image(img)
        cv2.imwrite(file_w_path, rect)

def crop_single_image(image):
    img_flipped = 255 * (image < 128).astype(np.uint8)
    coords = cv2.findNonZero(img_flipped)
    x, y, w, h = cv2.boundingRect(coords)
    rect = image[y:y + h, x:x + w]
    return rect

def find_contour_left_to_right(contour):
    bound = cv2.boundingRect(contour)
    return bound[0]

def seperate_lines_into_words():
    root = os.path.dirname(__file__) + "/../Data/lines/"
    for file_w_path in get_all_data_png(root):
        new_image_path = file_w_path.replace(".png", "")
        os.mkdir(new_image_path)
        print("Working on file: " + file_w_path)
        img = cv2.imread(file_w_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img, 150,255,cv2.THRESH_BINARY_INV)
        transform_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        dilated = cv2.dilate(thresh, transform_kernel,iterations=13)
        _, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours.sort(key=lambda x: find_contour_left_to_right(x))
        img_index = 0
        for contour in contours:
            [x, y, w, h] = cv2.boundingRect(contour)
            if h>500 and w>500:
                continue
            if h<40 or w<40 :
                continue
            img_index += 1
            roi = img[y:y+h, x:x+w]
            rect = crop_single_image(roi)
            cv2.imwrite(new_image_path + "/" + str(img_index) +  ".png", rect)

def get_all_data_png(root):
    list_png_paths = []
    for path, subdir, files in os.walk(root):
        for name in files:
            if "png" in name:
                name_w_path = os.path.join(path, name)
                list_png_paths.append(name_w_path)
    return list_png_paths


if __name__ == "__main__":
    seperate_lines_into_words()