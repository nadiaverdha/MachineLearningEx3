import import_cifar as ic
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3

import json
from keras.utils import plot_model
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

tf.random.set_seed(30)

if __name__ == '__main__':
    X_train, y_train, X_test, y_test, label_names = ic.load_train_test_data()
    # show_sample(label_names=label_names)
    y_train, y_test = ic.encode_y(y_train, y_test)  # perform OneHotEncoding
    # show_sample(label_names=label_names, sample_nr=5000)
    X_train = X_train/255.
    X_test = X_test/255.

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=30)

    X_train = tf.image.resize(X_train, (192, 192))  # reshaped to 6*32 size (min 75 is necessary for Inception)
    X_val = tf.image.resize(X_val, (192, 192))
    X_test = tf.image.resize(X_test, (192, 192))

    # prepare checkpoints

    cwd = os.getcwd()

    weights_path = os.path.join(cwd, 'models', 'cifar', 'saved_weights')
    try:
        os.makedirs(weights_path)
        print("Directory 'saved_weights' created successfully")
    except OSError as error:
        print("Directory 'saved_weights' can not be created - it might exist already")

    # early stopping to avoid overfitting/save computational power -> test set is used for validation
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    # ------------------------------------------------
    # MODEL transfer learning
    model_name = 'model_transfer_learning'
    model_dir = os.path.join(cwd, 'models', 'cifar', model_name)
    # checkpoints to save weights
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weights_path, model_name, "weights.{epoch:02d}-{val_loss:.2f}.h5"),  # must be set for each model
        save_weights_only=True,
        save_best_only=True)
    # check if model exists:
    if os.path.exists(model_dir):
        print("The model exists. Loading...")
        model = keras.models.load_model(model_dir)  # load model
    else:
        print("The model does not exist. Creating and fitting a new model...")

        model = keras.models.Sequential()
        base_model = InceptionV3(weights='imagenet', include_top=False,
                                 input_shape=(192, 192, 3))  # exclude top layers and last global pooling layer
        # %%
        # combine models
        model.add(base_model)
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(10, activation='softmax'))
        # %%
        weights_before_transfer = model.get_weights()
        base_model.trainable = False  # might not work -> see other book
        model.summary()

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        history = model.fit(X_train, y_train, epochs=400, batch_size=32,
                            validation_data=(X_test, y_test),
                            callbacks=[checkpoint_cb, early_stopping_cb])
        model.save(model_dir, save_format='tf')
        # save history for later plots - for loading history see test_cifar_tf_models.ipynb
        history_path = os.path.join(os.getcwd(), 'models', 'cifar', "history_" + model_name +".json")
        with open(history_path, "w") as f:
            json.dump(history.history, f)




    y_hat = pd.DataFrame({'predicted': model.predict(X_test).argmax(1), 'true': y_test.argmax(axis=1)})
    y_hat['predicted_label'] = ic.label_prediction(y_hat['predicted'], label_names)
    y_hat['true_label'] = ic.label_prediction(y_hat['true'], label_names)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('---' * 40)
    print('Test loss: {:.3f}'.format(test_loss))
    print('Test accuracy: {:.3f}'.format(test_acc))