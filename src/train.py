import cv2
from imutils import paths

def load_imgs_and_paths(labels):
    for label in labels:

        image_dir = 'data/{}'.format(label)
        print('Loading from {}'.format(image_dir))

        for path in paths:


            paths.list_images('data/{}'.format(label))


if __name__ == '__main__':

    labels = ['dunk', 'jumpshot']
    load_imgs_and_paths(labels)