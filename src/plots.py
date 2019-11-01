import os
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

def get_image(path):
    img = imread(path)
    img = resize(img, (224,224))
    return img

def get_all_images(dir):
    images = []
    for filename in os

def plot_image(v, ax, title):
    # img = v.reshape(224,244)
    ax.imshow(img)
    ax.set_title(title)
    plt.savefig('plots/{}'.format(title))
    plt.show()
    return ax

def plot_avg(mat, ax, catergory):

    avg_img = mat.mean(axis = 0)
    plot_image(avg_img, ax, 'Average {}'.format(category))
    return ax

if __name__ == '__main__':
    path = '/home/lucas/Galvanize/Projects/Capstone-3/data/google_imgs/test_dunk_img/1.919689104.jpg'

    img = get_image(path)
    print(img)
    fig, ax = plt.subplots(1,1)
    plot_image(img, ax, 'dunk!')
    plt.show()