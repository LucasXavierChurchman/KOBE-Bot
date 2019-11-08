import os
from tempfile import TemporaryFile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from skimage import exposure
from skimage.color import gray2rgb, rgb2gray, rgba2rgb
from skimage.io import imread, imread_collection
from skimage.transform import resize
from sklearn.decomposition import PCA

from image_processing import get_all_images, get_image, images_to_array


def plot_image_color(v, ax, resolution, title):
    '''
    Displays a full color image with a given resolution
    '''
    img = resize(v, (resolution, resolution))
    ax.imshow(img)
    ax.set_title(title)
    # plt.savefig('../plots+images/{}'.format(title))
    # plt.show()
    return ax

def plot_image_gray(v, ax, resolution, title, heatmap = False):
    img = resize(v, (resolution, resolution))
    if heatmap:
        ax.imshow(rgb2gray(img))
        ax.set_title(title)
        # plt.savefig('../plots+images/{}'.format(title))
        # plt.show()
        return ax
    else:
        img = exposure.equalize_hist(img, nbins=600)
        ax.imshow(rgb2gray(img), cmap= plt.cm.gray)
        ax.set_title(title)
        # plt.savefig('../plots+images/{}'.format(title))
        # plt.show()
        return ax

def plot_intensities(v, ax, resolution):
    img = resize(v, (resolution,resolution))
    ax.hist(resize(img, (240,240)).ravel(), bins = 100, color = 'Red', alpha = 0.6)
    ax.set_title('Pixel Intensities')
    ax.set_xlim(right =1)
    # ax.set_ylim(top = 7001)
    return ax

def plot_processing_demo(image_path, ax_color, ax_heatmap, ax_gray, title):
    resolutions = [240, 240, 240]
    img = get_image(image_path)
    img = np.array(img)
    ax_color = plot_image_color(img, 
                                ax_color, 
                                resolutions[0], 
                                '{} Color\n{}x{}'.format(title, resolutions[0], resolutions[0])
                            )   
    ax_heatmap = plot_image_gray(img, 
                                ax_heatmap, resolutions[1], 
                                '{} Grayscale\n{}x{}'.format(title, resolutions[1], resolutions[1]),
                                heatmap = True
                            )
    ax_gray = plot_image_gray   (img, 
                                ax_gray, resolutions[2], 
                                '{} Grayscale High Contrast\n{}x{}'.format(title, resolutions[2], resolutions[2])
                                )

def plot_avgs(image_array_path, ax_color, ax_gray, ax_hist, category):
    '''
    Generates a plot
    '''
    resolutions = [240, 240, 240]
    image_array = np.load(image_array_path)
    print(image_array.shape)
    avg_img = image_array.mean(axis = 0)
    print(avg_img.shape)


    ax_color = plot_image_color(avg_img, 
                                ax_color, 
                                resolutions[0], 
                                '{} Color\n{}x{}'.format(category, resolutions[0], resolutions[0])                           )   

    ax_gray = plot_image_gray(  avg_img, 
                                ax_gray, 
                                resolutions[1], 
                                '{} Grayscale High Contrast\n{}x{}'.format(category, resolutions[2], resolutions[2])
                                )

    ax_hist = plot_intensities( avg_img,
                                ax_hist,
                                resolutions[2]
                                )

    return ax_color, ax_gray, ax_hist

def plot_pca(image_array_path, axs1, ax2, ax3, ax4, ax5):
    '''
    Plots top 5 'eigen images' for a set of images. Lighter colored areas in these eigenimages indicate
    areas of that are more deterministic/unique per image in the set.
    '''
    axs = [axs1, ax2, ax3, ax4, ax5]
    image_array = np.load(image_array_path)
    flat_array = []
    for img in image_array:
        img = rgb2gray(img)
        img = pd.Series(img.flatten())
        flat_array.append(img)
    flat_array = np.array(flat_array)
    pca = PCA(n_components=0.5)
    pca.fit(flat_array)
    for i, ax in enumerate(axs):
        ax.imshow(pca.components_[i].reshape(240,240),cmap = plt.cm.gray)
        ax.grid(False)
    return axs

def plot_model_results(model_path):
    model = load_model(model_path)

    print(model.history.history)

if __name__ == '__main__':
    matplotlib.style.use('ggplot')

    plot_model_results('../models/broadcast_100_epochs_88_acc.model')

    # fig, axs = plt.subplots(2,5, figsize=(12,4))
    # plot_pca('../data/image_arrays/broadcast_denver_dunk.npy', axs[0,0], axs[0,1], axs[0,2], axs[0,3], axs[0,4])
    # plot_pca('../data/image_arrays/broadcast_denver_three.npy', axs[1,0], axs[1,1], axs[1,2], axs[1,3], axs[1,4])
    # rows = ['Jumpshot','Dunk']
    # for ax, row in zip(axs[:,0], rows):
    #     ax.set_ylabel(row, rotation=90, size='large')
    # plt.tight_layout()
    # plt.savefig('../plots+images/denver_pca')

    # # path = '../data/google_imgs/test_jumpshot/1.maxresdefault.jpg'
    # # img = get_image(path)
    # fig, axs = plt.subplots(2,3, figsize=(12,8))

    # jumpshot_image_list = get_all_images('../data/google_imgs/jumpshot')
    # jumpshot_image_array = images_to_array(jumpshot_image_list, 'jumpshot')
    # fig, axs = plt.subplots(2,3, figsize=(12,8))
    # plot_avgs('../data/image_arrays/broadcast_denver_three.npy', axs[0,0], axs[0,1], axs[0,2], '')
    # plot_avgs('../data/image_arrays/broadcast_denver_dunk.npy', axs[1,0], axs[1,1], axs[1,2], '')
    # rows = ['Jump Shot','Dunk']
    # for ax, row in zip(axs[:,0], rows):
    #     ax.set_ylabel(row, rotation=90, size='large')
    # plt.tight_layout()
    # plt.savefig('../plots+images/denver_image_avgs')
    # plt.show()

    # fig, axs = plt.subplots(2,3, figsize=(12,8))
    # plot_avgs('../data/image_arrays/jumpshot.npy', axs[0,0], axs[0,1], axs[0,2], '')
    # plot_avgs('../data/image_arrays/dunk.npy', axs[1,0], axs[1,1], axs[1,2], '')
    # rows = ['Jump Shot','Dunk']
    # for ax, row in zip(axs[:,0], rows):
    #     ax.set_ylabel(row, rotation=90, size='large')
    # plt.tight_layout()
    # plt.savefig('../plots+images/google_image_avgs')
    # plt.show()

    # # dunk_image_list = get_all_images('../data/google_imgs/dunk')
    # # dunk_image_array = images_to_array(dunk_image_list, 'jumpshot')
    # plot_avgs('../data/dunk.npy', axs[1,0], axs[1,1], axs[1,2], 'Dunk')

    # fig.suptitle('Average Jumpshot vs Dunk, All Images')
    # # plt.tight_layout()
    # plt.savefig('../plots+images/avg_google_img_all.png')

    # fig, axs = plt.subplots(1,3, figsize=(12,4))
    # plot_avgs('../data/image_arrays/jamal_array.npy', axs[0], axs[1], axs[2], '')
    # plt.savefig('../plots+images/single_image_processing.png')


    # all_area_dunk = np.load('../data/image_arrays/all_arena_dunk.npy') 
    # fig, ax = plt.subplots( 1,1, figsize = (4,4))
    # plot_image_gray(np.mean(all_area_dunk, axis = 0), ax, 240, 'Example of Average Dunk Image\nfrom Several Arenas')
    # plt.tight_layout()
    # plt.savefig('../plots+images/all_arena_example')
