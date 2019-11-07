import os
from skimage.io import imread, imread_collection
from skimage.transform import resize
from skimage.color import rgb2gray, rgba2rgb, gray2rgb
from tempfile import TemporaryFile
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def get_image(path):
    img = imread(path, plugin='matplotlib')
    img = resize(img, (240,240))
    return img

def get_all_images(folder_path):
    image_list = []
    for filename in os.listdir(folder_path):
        print(filename)
        img = get_image(os.path.join(folder_path, filename))
        image_list.append(img)
    return image_list

def images_to_array(image_list, save_name):
    for n, img in enumerate(image_list):
        if len(img.shape) == 2: #if grayscale convery to rgb
            # image_list[n] == gray2rgb(img)
            image_list.pop(n) #gray2rgb is deprecated and can't find a solutuion. Will pop in the meantime
        if len(img.shape) == 3 and img.shape[2] == 4: #if rgba convert to rgb
            image_list[n] = rgba2rgb(img)
    image_array = np.array(image_list)
    np.save('../data/image_arrays/{}'.format(save_name), image_array)
    return image_array

if __name__ == '__main__':
    pass