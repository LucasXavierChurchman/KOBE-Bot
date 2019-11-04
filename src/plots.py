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
    np.save('../data/{}'.format(save_name), image_array)
    return image_array

def plot_image_color(v, ax, resolution, title):

    img = resize(v, (resolution, resolution))
    ax.imshow(img)
    ax.set_title(title)
    # plt.savefig('../plots/{}'.format(title))
    # plt.show()
    return ax

def plot_image_heatmap(v, ax, resolution, title):

    img = resize(v, (resolution, resolution))
    ax.imshow(rgb2gray(img))
    ax.set_title(title)
    # plt.savefig('../plots/{}'.format(title))
    # plt.show()
    return ax

def plot_image_gray(v, ax, resolution, title):

    img = resize(v, (resolution, resolution))
    ax.imshow(rgb2gray(img), cmap=plt.cm.gray)
    ax.set_title(title)
    # plt.savefig('../plots/{}'.format(title))
    # plt.show()
    return ax

def plot_avgs(image_array_path, ax_color, ax_heatmap, ax_gray, category):

    resolutions = [240, 50, 25]
    image_array = np.load(image_array_path)
    avg_img = image_array.mean(axis = 0)
    ax_color = plot_image_color(avg_img, ax_color, resolutions[0], '{} Color {}x{}'.format(category, resolutions[0], resolutions[0]))
    ax_heatmap = plot_image_heatmap(avg_img, ax_heatmap, resolutions[1], '{} Grayscale Heatmap {}x{}'.format(category, resolutions[1], resolutions[1]))
    ax_gray = plot_image_gray(avg_img, ax_gray, resolutions[2], '{} Grayscale {}x{}'.format(category, resolutions[2], resolutions[2]))

    return ax_color, ax_heatmap, ax_gray

if __name__ == '__main__':

    # path = '../data/google_imgs/test_jumpshot/1.maxresdefault.jpg'
    # img = get_image(path)
    fig, axs = plt.subplots(3,2)

    # jumpshot_image_list = get_all_images('../data/google_imgs/jumpshot')
    # jumpshot_image_array = images_to_array(jumpshot_image_list, 'jumpshot')
    plot_avgs('../data/jumpshot.npy', axs[0,0], axs[1,0], axs[2,0], 'Jumpshot')

    # dunk_image_list = get_all_images('../data/google_imgs/dunk')
    # dunk_image_array = images_to_array(dunk_image_list, 'jumpshot')
    plot_avgs('../data/dunk.npy', axs[0,1], axs[1,1], axs[2,1], 'Dunk')

    plt.title('Average Jumpshot vs Dunk, 10 Images')
    plt.tight_layout()
    plt.savefig('../plots/avg_google_img_all.png')
    plt.show()

