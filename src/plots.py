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
from sklearn.decomposition import NMF, PCA

from image_processing import get_all_images, get_image, images_to_array

def plot_image_color(v, ax, resolution, title):
    '''
    Displays a full color image from an image array with a given resolution

    Arguments:
    v: numpy image array
    ax: matplotlib axis
    resoltion: square size for the output image plot
    title: axis title

    Returns: matplotlib axis
    '''
    img = resize(v, (resolution, resolution))
    ax.imshow(img)
    ax.set_title(title)
    return ax

def plot_image_gray(v, ax, resolution, title, greenblue = False):
    '''
    Displays a grayscale image from an image array with a given resoltuon

    Arguments:
        v: numpy image array
        ax: matplotlib axis
        resoltion: square size for the output image plot
        title: axis title
        greenblue: If True the plot will be plotted without the grayscale colormap

    Returns: matplotlib axis
    '''
    img = resize(v, (resolution, resolution))
    if greenblue:
        ax.imshow(rgb2gray(img))
        ax.set_title(title)
        return ax
    else:
        ax.imshow(rgb2gray(img), cmap= plt.cm.gray)
        ax.set_title(title)
        return ax

def plot_intensities(v, ax, resolution):
    '''
    Plots a pixel intensity histogram from an image array at a given resolution

    Arguments:
        v: numpy image array
        ax: matplotlib axis
        resoltion: square size for the output image plot

    Returns: matplotlib axis
    '''
    img = resize(v, (resolution,resolution))
    ax.hist(resize(img, (240,240)).ravel(), bins = 100, color = 'Red', alpha = 0.6)
    ax.set_title('Pixel Intensities')
    ax.set_xlim(right =1)
    # ax.set_ylim(top = 7001)
    return ax

def plot_processing_demo(path0, path1, path2, path3, ax0, ax1, ax2, ax3):
    '''
    Plots 4 images. Used to display each type and class combination of images

    Arguments:
        path0, path1, path2, path3: paths to the image files that will be plotted
        ax0, ax1, ax2, ax3: axes to be plotted on

    Returns: ax0, ax1, ax2, ax3
    '''
    resolutions = [240, 240, 240]
    img0 = get_image(path0)
    img0 = np.array(img0)
    ax0 = plot_image_color(img0 ,ax0, resolutions[0], 'Google Image Jumpshot')

    img1 = get_image(path1)
    img1 = np.array(img1)
    ax1 = plot_image_color(img1 ,ax1, resolutions[0], 'Google Image Dunk')

    img2 = get_image(path2)
    img2 = np.array(img2)
    ax2 = plot_image_color(img2 ,ax2, resolutions[0], 'Broadcast Angle Jumpshot')

    img3 = get_image(path3)
    img3 = np.array(img3)
    ax3 = plot_image_color(img3 ,ax3, resolutions[0], 'Broadcast Angle Dunk')

    return ax0, ax1, ax2, ax3


def plot_avgs(v_path, ax_color, ax_gray, ax_hist, category):
    '''
    Generates plots of an image array in color, grayscale, and its pixel intensity histogram
    using predefined plotting functions. Loads in a saved numpy array since the arrays
    for entire classes take a while to generate

    Arguments:
        v_path: path to image array. 
        ax_color, ax_gray, ax_hist: axes for color, grayscale, and pixel intensity plots
        category: image category type for axis title

    returns: ax_color, ax_gray, ax_hist
    '''
    resolutions = [240, 240, 240]
    image_array = np.load(v_path)
    avg_img = image_array.mean(axis = 0)

    ax_color = plot_image_color(avg_img, 
                                ax_color, 
                                resolutions[0], 
                                '{} Color\n{}x{}'.format(category, resolutions[0], resolutions[0])                           )   

    ax_gray = plot_image_gray(  avg_img, 
                                ax_gray, 
                                resolutions[1], 
                                '{} Grayscale\n{}x{}'.format(category, resolutions[2], resolutions[2])
                                )
    ax_hist = plot_intensities( avg_img,
                                ax_hist,
                                resolutions[2]
                                )

    return ax_color, ax_gray, ax_hist

def plot_nmf(v_path, ax0, ax1, ax2, ax3, ax4):
    '''
    Plots NMF decompositions for a set of images. Lighter colored areas in these eigenimages indicate
    areas of that are more deterministic/unique per image in the set.

    Arguments:
        v_path: path to image array.
        ax0, ax1, ax2, ax3, ax4: matplotlib axes for each eigen image

    returns: list of axes
    '''
    axs = [ax0, ax1, ax2, ax3, ax4]
    image_array = np.load(v_path)
    flat_array = []
    for img in image_array:
        img = rgb2gray(img)
        img = pd.Series(img.flatten())
        flat_array.append(img)
    flat_array = np.array(flat_array)
    nmf = NMF(n_components=10)
    nmf.fit(flat_array)
    for i, ax in enumerate(axs):
        print(nmf.components_[i])
        ax.imshow(nmf.components_[i].reshape(240,240),cmap = plt.cm.gray)
        ax.grid(False)
    return axs

def plot_model_results(model_history_path, ax_loss, ax_acc):
    '''
    Plots a model's accuracy and loss from its history csv

    Arguments: 
        model_history_path: path to model history csv
        ax_loss, ax_acc: axes for loss and accuracy plots
    '''
    history = pd.read_csv(model_history_path)

    ax_loss.plot(history.loss, 
                color = 'orangered', 
                label = 'Loss', 
                alpha = 0.6)
    ax_loss.plot(history.val_loss, 
                color = 'purple', 
                label = 'Validation Loss', 
                alpha = 0.6)
    ax_loss.annotate(np.round(history.loss.iloc[-1], 2), 
                    xy = (200, history.loss.iloc[-1]))
    ax_loss.annotate(np.round(history.val_loss.iloc[-1], 2), 
                    xy = (200, history.val_loss.iloc[-1]))      
    ax_loss.legend()          

    ax_acc.plot(history.accuracy, 
                color = 'orangered', 
                label = 'Accuracy', 
                alpha = 0.6)
    ax_acc.plot(history.val_accuracy, 
                color = 'purple', 
                label = 'Validation Accuracy', 
                alpha = 0.6)
    ax_acc.annotate(np.round(history.accuracy.iloc[-1], 2), 
                    xy = (200, history.accuracy.iloc[-1]))
    ax_acc.annotate(np.round(history.val_accuracy.iloc[-1], 2), 
                    xy = (200, history.val_accuracy.iloc[-1]))
    ax_acc.legend()
    
    print(history)

if __name__ == '__main__':
    matplotlib.style.use('ggplot')

    fig, axs = plt.subplots(2,5, figsize=(12,4))

    plot_nmf('../data/image_arrays/denver_jumpshot.npy', axs[0,0], axs[0,1], axs[0,2], axs[0,3], axs[0,4])
    plot_nmf('../data/image_arrays/denver_dunk.npy', axs[1,0], axs[1,1], axs[1,2], axs[1,3], axs[1,4])

    rows = ['Jumpshot','Dunk']
    for ax, row in zip(axs[:,0], rows):
        ax.set_ylabel(row, rotation=90, size='large')
    plt.tight_layout()
    plt.show()
