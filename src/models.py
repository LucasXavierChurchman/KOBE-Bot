import random
import cv2
import cv2.cv2 as cv2 #extra import gets rid of error warnings
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers.pooling import AveragePooling2D
from keras.optimizers import SGD
from keras.layers import Input
from keras.models import Model


def load_images_and_labels(target_labels):
    '''
    Loads all images from directories with names in target_labels
    a label and image array

    TO DO: Make this work with np.load() and skimage (if possible)
    '''
    images = []
    labels = []

    for label in target_labels:
        image_dir = '../data/google_imgs/{}'.format(label)
        print('Loading from {}'.format(image_dir))
        image_paths = list(paths.list_images(image_dir))

        for path in image_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (240, 240))
            paths.list_images('../data/google_imgs/{}'.format(label))

            images.append(img)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def train_CNN(images, labels, epochs, savename):

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, 
                                                        test_size = 0.20,
                                                        stratify = labels,
                                                        random_state = 17)

    ImageNet_mean = np.array([ 123.68, 116.779, 103.939 ])

    train_transformations = ImageDataGenerator(
                        rotation_range=45,
                        zoom_range=0.25,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.15,
                        horizontal_flip=True,
                        fill_mode="wrap") #constant, nearest, reflect or wrap

    validation_transformations = ImageDataGenerator(ImageNet_mean)  

    train_transformations.mean = ImageNet_mean
    validation_transformations.mean = ImageNet_mean   

    #load transferred learning model. Need to try resnet50 as well
    transferred_model = ResNet50(weights = 'imagenet',
                                include_top = False,
                                input_tensor= Input(shape=(240, 240, 3)))

    #build head model
    head_model = transferred_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(512, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(len(lb.classes_), activation="softmax")(head_model)

    model = Model(inputs=transferred_model.input, outputs=head_model)

    #freeze layers from transferred model
    for layer in transferred_model.layers:
	    layer.trainable = False

    opt = SGD(lr=0.0001, momentum=0.9, decay=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    #need 2 column array for target
    y_train = np.hstack((y_train, 1-y_train))
    y_test = np.hstack((y_test, 1-y_test))

    #gotta figure out why this wont work
    # H = model.fit_generator(
    #                         train_transformations.flow(X_train, y_train, batch_size=32),
    #                         steps_per_epoch=len(X_train) // 32,
    #                         validation_data=validation_transformations.flow(X_test, y_test),
    #                         validation_steps=len(X_test) // 32,
    #                         epochs=epochs)

    model.fit(train_transformations.flow(X_train, y_train), 
            validation_data = (X_test, y_test),
            epochs = epochs)



    return model
 
if __name__ == '__main__':
    random.seed(17)
    savename = 'dunk_v_shot'
    epochs = 10
    target_labels = ['dunk', 'jumpshot']
    images, labels = load_images_and_labels(target_labels)
    model = train_CNN(images, labels, epochs = epochs, savename = savename)
    model.save('../models/{}_{}_epochs_grayscale.model'.format(savename, epochs))