from imutils import contours
from skimage import measure
from parse_text_file_data import *
import sklearn
import PIL.Image
import sys
import cv2
import os.path
import numpy as np
import imutils
import time
import multiprocessing as mp

class image:
    def __init__(self, img, image_name, image_index=None):
        self.img = img
        self.image_name = image_name
        self.image_index = image_index

def convert_images_to_bw(image):
    """
    Converts an image to black and white 0 or 255 RGB values
    :param image: cv2 image object that is to be converted to black and white
    :return: returns the processed black and white image
    """
    (thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw

def make_all_images_bw():
    """
    this does through all the images in the data set and makes them all white and black. This is to help with processing
    the data for feeding into the learner.
    """
    root = os.path.dirname(__file__) + "/../Data/"
    for name_w_path in get_all_data_png(root):
        im_gray = cv2.imread(name_w_path, cv2.IMREAD_GRAYSCALE)
        im_bw = convert_images_to_bw(im_gray)
        cv2.imwrite(name_w_path, im_bw)

def crop_data_all_images(data_type):
    """
    Crops all of the images so that there is no extra space surrounding the black pixels.
    :param data_type: word, sentence, or line. These are the three datat ypes in the data set.
    """
    root = os.path.dirname(__file__) + "/../Data/" + str(data_type)
    for file_w_path in get_all_data_png(root):
        img = cv2.imread(file_w_path,cv2.IMREAD_GRAYSCALE)
        rect = crop_single_image(img)
        cv2.imwrite(file_w_path, rect)

def crop_single_image(image):
    """
    Crops a single image of all the extra white space around the blak pixels in the image
    :param image: cv2 image object that is to be cropped
    :return: returns the cropped image object after it has been processed
    """
    img_flipped = 255 * (image < 128).astype(np.uint8)
    coords = cv2.findNonZero(img_flipped)
    x, y, w, h = cv2.boundingRect(coords)
    rect = image[y:y + h, x:x + w]
    return rect

def find_contour_left_to_right(contour):
    """
    Helper function to sort the contours of a sentence or line data type so that they can be split into words in
    seperate_lines_into_words
    :param contour: contour object
    :return: returns the right most point of the contour so we can sort left to right
    """
    bound = cv2.boundingRect(contour)
    return bound[0]

def seperate_lines_into_words():
    """
    Sepeartes all the line date type images in the data set into words and then stores them in directories that match
    the names of the original image. Inside the new directory, all of the words will be names 1.png.... n.png.
    """
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
            word = img[y:y+h, x:x+w]
            rect = crop_single_image(word)
            cv2.imwrite(new_image_path + "/" + str(img_index) +  ".png", rect)

def find_largest_image(data_type):
    """
    Finds the largest image in the data set. Has to have the largest width and height
    :param data_type: can be line,sentence, or word
    :return: the height and width of the largest image
    """
    root = os.path.dirname(__file__) + "/../Data/" + data_type + "/"
    largest = np.zeros((1,1,1),np.uint8)
    for file_w_path in get_all_data_png(root):
        img = cv2.imread(file_w_path)
        if img.shape[0] > largest.shape[0] and img.shape[1] > largest.shape[1]:
            largest = img
    print("This is the largest image by height " + str(largest.shape[0]) + " and width " + str(largest.shape[1]))
    return largest.shape[0], largest.shape[1]

def find_smallest_image(data_type):
    """
    Finds the smallest image in the data set. Has to have the smallest width and height
    :param data_type: can be line,sentence, or word
    :return: the height and width of the smallest image
    """
    root = os.path.dirname(__file__) + "/../Data/" + data_type + "/"
    smallest = np.zeros((10000, 10000, 1), np.uint8)
    for file_w_path in get_all_data_png(root):
        img = cv2.imread(file_w_path)
        if img.shape[0] < smallest.shape[0] and img.shape[1] < smallest.shape[1]:
            smallest = img
    print("This is the smallest image by height " + str(smallest.shape[0]) + " and width " + str(smallest.shape[1]))
    return smallest.shape[0], smallest.shape[1]

def get_all_data_png(root,containing = None):
    """
    Grabs the path of all the png images in the data set under a specified root
    :param root: The roor directory in which to start the search for all the images in the data set.
    :return:
    """
    list_png_paths = []
    for path, subdir, files in os.walk(root):
        for name in files:
            if containing == None:
                if "png" in name:
                    name_w_path = os.path.join(path, name)
                    list_png_paths.append(name_w_path)
            else:
                if "png" in name and containing in name:
                    name_w_path = os.path.join(path, name)
                    list_png_paths.append(name_w_path)
    return list_png_paths

def resize_images(data_type,scale="n"):
    """
    Resize all of the images of a specific data_type so that they will be the same size of either the smallest or the largest image in the data set.
    :param data_type: Can be line, sentence, or word
    :param scale: "d" is for downscale and "u" is for upscale
    :return:
    """
    if scale == "d":
        #downscale all the images so that they are the same size as the smallest image in the data set
        #trying to test is downscaling, upcaling, or no scaling results in best ml performance
        height, width = find_smallest_image(data_type)
    elif scale == "u":
        #option upscales all the images in the data set to match those of the largest image in the data set
        height, width = find_largest_image(data_type)
    elif scale == "n":
        height, width = 32, 128
    root = os.path.dirname(__file__) + "/../Data/" + data_type + "/"
    for file_w_path in get_all_data_png(root):
        try:
            png_file_name = file_w_path.split("/")[-1]
            print(png_file_name)
            #make sure to only downsize the images that were sepearted from lines/sentences to words
            if len(png_file_name) < 10 or data_type == "words":
                img = cv2.imread(file_w_path)
                resized_img = cv2.resize(img,(width,height), interpolation=cv2.INTER_NEAREST)
                png_directory = file_w_path.replace(png_file_name,"")
                file_name = png_file_name.replace(".png","")
                path_to_write = png_directory+file_name+"_resized_"+scale+".png"
                if os.path.isfile(path_to_write):
                    os.remove(path_to_write)
                cv2.imwrite(path_to_write, resized_img)
            else:
                continue
        except Exception as e:
            print("Unable to resize " + file_name)
            continue

def delete_resied_images(data_type):
    root = os.path.dirname(__file__) + "/../Data/" + data_type +  "/"
    print(root)
    for file_w_path in get_all_data_png(root):

        png_file_name = file_w_path.split("/")[-1]
        if "resized" in png_file_name:
            print("deleting file: " + str(file_w_path))
            os.remove(file_w_path)


def convert_resized_images_to_numpy(data_type,start=0,data_size=None):
    """
    Converts the upscaled images into a single numpy array and contructs the y_train array for feeding into tensor flow learner
    :param data_type: Select one if the 3 data_types that you would like to convert into numpy array
    :param data_size: how large should the data set be
    :return:
    """

    root = os.path.dirname(__file__) + "/../Data/" + data_type + "/"
    list_resized_images = get_all_data_png(root,"resized")
    if data_size == None:
        data_size = len(list_resized_images)
    pool_outputs = mulprocess_image_to_np(list_resized_images,start,data_size)
    y_train_dict  = read_text_data("lines.txt")
    y_train = []
    x_train = []
    first_image = pool_outputs[0]
    x_train.append(first_image.img)
    y_train.append(y_train_dict.get(first_image.image_name)[int(first_image.image_index)])
    for img_obj in pool_outputs[1:]:
        try:
            y_train.append(y_train_dict.get(img_obj.image_name)[int(img_obj.image_index) - 1])
            x_train.append(img_obj.img)
        except Exception as e:
            print("skipped because of error" + str(e))
            print("Skipped image: " + str(img_obj.image_name) + " ,index: " +str(int(img_obj.image_index) - 1))
            continue
    y_train = np.asarray(y_train)
    x_train = np.asarray(x_train)
    return x_train,y_train

def mulprocess_image_to_np(image_list,start,data_size):
    pool_size = mp.cpu_count()
    pool = mp.Pool(processes=pool_size)
    list_np_arrays = pool.map(process_image_to_np, image_list[start:data_size])
    pool.close()
    pool.join()
    return list_np_arrays

def flatten_images(list_images_path):
    img = cv2.imread(list_images_path,cv2.IMREAD_GRAYSCALE)
    x_train = img.flatten()
    flattened_img = image(x_train,list_images_path.split("/")[-2], list_images_path.split("/")[-1].split("_")[0])
    return flattened_img

def process_image_to_np(list_images_path):
    img = cv2.imread(list_images_path, cv2.IMREAD_GRAYSCALE)
    #normalize images
    np_img = image(img/255, list_images_path.split("/")[-1])
    return np_img

def get_words(start=0,data_size=None):
    root = os.path.dirname(__file__) + "/../Data/words/"
    list_words = get_all_data_png(root,"resized")
    pool_outputs = mulprocess_image_to_np(list_words,start,data_size)
    y_train_dict = read_word_test_data("words.txt")
    y_train = []
    x_train = []
    first_image = pool_outputs[0]
    x_train.append(first_image.img)
    y_train.append(y_train_dict.get(str(first_image.image_name).split("_")[0]))
    for img_obj in pool_outputs[1:]:
        try:
            y_train.append(y_train_dict.get(str(img_obj.image_name).split("_")[0]))
            x_train.append(img_obj.img)
        except Exception as e:
            print("skipped because of error" + str(e))
            print("Skipped image: " + str(img_obj.image_name) )
            continue
    y_train = np.asarray(y_train)
    x_train = np.asarray(x_train).reshape(len(pool_outputs),32,128,1)
    print(y_train)
    return x_train, y_train
    

if __name__ == "__main__":
    delete_resied_images("words")


