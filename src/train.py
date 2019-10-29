import cv2
import numpy as np
from imutils import paths

def load_images_and_paths(target_labels):
    images = []
    labels = []

    for label in target_labels:
        image_dir = 'data/{}'.format(label)
        print('Loading from {}'.format(image_dir))
        image_paths = list(paths.list_images(image_dir))

        for img in image_paths:
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            paths.list_images('data/{}'.format(label))

            images.append(img)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels




if __name__ == '__main__':

    labels = ['dunk', 'jumpshot']
    images, labels = load_images_and_paths(labels)