"""
File for importing data (cifar and monkeys) from local
http://www.cs.toronto.edu/~kriz/cifar.html
https://www.kaggle.com/datasets/slothkong/10-monkey-species?resource=download
Folder structure should be data -> cifar | data -> monkeys
cifar: \MachineLearningEx3\data\cifar (must be prepared)
monkeys: MachineLearningEx3\data\monkeys\training\training
         MachineLearningEx3\data\monkeys\validation\validation
"""
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder


def unpickle(file):
    """
    returns a  from cifar data
    :param file:
    :return:
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
#%% md

#%%
def load_train_test_data(path_to_data='data/cifar/cifar-10-batches-py'):
    """
    :param path_to_data: path to
    :return: X_train, y_train, X_test, y_test, label_names
    """

    dict_b1 = unpickle(path_to_data + "/data_batch_1")
    dict_b2 = unpickle(path_to_data + "/data_batch_2")
    dict_b3 = unpickle(path_to_data + "/data_batch_3")
    dict_b4 = unpickle(path_to_data + "/data_batch_4")
    dict_b5 = unpickle(path_to_data + "/data_batch_5")
    dict_meta = unpickle(path_to_data + "/batches.meta")
    dict_test = unpickle(path_to_data + "/test_batch")

    data_b1 = dict_b1[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2,3,1)
    data_b2 = dict_b2[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2,3,1)
    data_b3 = dict_b3[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2,3,1)
    data_b4 = dict_b4[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2,3,1)
    data_b5 = dict_b5[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2,3,1)
    labels_b1 = dict_b1[b'labels']
    labels_b2 = dict_b2[b'labels']
    labels_b3 = dict_b3[b'labels']
    labels_b4 = dict_b4[b'labels']
    labels_b5 = dict_b5[b'labels']


    X_test = dict_test[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2,3,1)
    y_test = np.array(dict_test[b'labels'])
    X_train = np.concatenate([data_b1, data_b2, data_b3, data_b4, data_b5])
    y_train = np.concatenate([labels_b1, labels_b2, labels_b3, labels_b4, labels_b5])
    label_names = dict_meta[b'label_names']
    label_names = [name.decode('utf-8') for name in label_names]  # encode (remove b'')

    return X_train, y_train, X_test, y_test, label_names


def encode_y(y_train, y_test):
    """
    performe OneHotEncoding on train and test targets
    :param y_train:
    :param y_test:
    :return:
    """
    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(y_train.reshape(-1,1))
    y_test_encoded = encoder.transform(y_test.reshape(-1,1))
    return y_train_encoded, y_test_encoded

    # Choose an index to select an image from the dataset
    index = 27001

def show_sample(label_names, sample_nr=None, random_max=50000):
    """
    show an (random) image.
    :param sample_nr: default: None -> random number is taken
    :param random_max: default=50000 (for train data)
    """
    if sample_nr is None:
        sample_nr = random.randint(0,random_max)

    # Set the figure size
    # plt.figure(figsize=(0.9, 0.9))

    # Get the image and label
    image = X_train[sample_nr]
    label = y_train[sample_nr]
    if isinstance(label, np.ndarray):  # for OneHotEncoded Data
        label = np.argmax(label)

    # Plot the image
    plt.imshow(image)
    plt.title(label_names[label])
    plt.axis('off')
    plt.show()

def label_prediction(arr, label_names):
    """
    function to write label names instead of encoded labels
    :param arr: predicted values [7, 2, 4 ,8, 0]
    :param label_names: list of label names in the correct order
    :return: array with labels [deer, dog, ship, cat, airplane]
    """

    label_mapping = dict(zip(range(len(label_names)), label_names))
    label_arr = [label_mapping[label] for label in arr]

    return label_arr



if __name__ == '__main__':

    X_train, y_train, X_test, y_test, label_names = load_train_test_data()
    # show_sample(label_names=label_names)
    y_train, y_test = encode_y(y_train, y_test)  # perform OneHotEncoding
    # show_sample(label_names=label_names, sample_nr=5000)

    X_train = X_train.astype('float') / 255
    X_test = X_test.astype('float') / 255


    show_sample(label_names=label_names, sample_nr=5000)
