import import_cifar as ic
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import json
from keras.utils import plot_model
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

tf.random.set_seed(30)


def model_CNN_S(input_shape):
    model = keras.models.Sequential()
    # conv_block 1
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                                  input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))  # pooling block 1
    # conv_block 2
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # prediction block
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

def model_CNN_M(input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same',
                                  input_shape=input_shape))  # stride is default 1
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))  # default size = 2
    model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # prediction block
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

def model_CNN_L(input_shape):
    model = keras.models.Sequential()
    # conv_block 1
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same',
                                        input_shape=input_shape))  # stride is default 1
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))  # default size = 2
    model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # prediction block
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

def model_CNN_XL(input_shape):
    model = keras.models.Sequential()
    # conv_block 1
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same',
                                        input_shape=input_shape))  # stride is default 1
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))  # default size = 2
    model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # prediction block
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, label_names = ic.load_train_test_data()
    # show_sample(label_names=label_names)
    y_train, y_test = ic.encode_y(y_train, y_test)  # perform OneHotEncoding
    # show_sample(label_names=label_names, sample_nr=5000)
    X_train = X_train.astype('float32') / 255
    X_test = X_test / 255.

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=30)

    input_shape = X_train.shape[1:]

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
    # MODEL CNN_S
    model_name = 'CNN_S'
    model_dir = os.path.join(cwd, 'models', 'cifar', 'model_' + model_name)
    # checkpoints to save weights
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weights_path, model_name, "weights.{epoch:02d}-{val_loss:.2f}.h5"),  # must be set for each model
        save_weights_only=True,
        save_best_only=True)
    print('---'*40)
    print('---' * 40)
    print(model_name + ':')
    # check if model exists:
    if os.path.exists(model_dir):
        print("The model exists. Loading...")
        model_CNN_S = keras.models.load_model(model_dir)  # load model
    else:
        print("The model does not exist. Creating and fitting a new model...")
        model_CNN_S = model_CNN_S(input_shape)

        model_CNN_S.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

        history_CNN_S = model_CNN_S.fit(X_train, y_train, epochs=400, batch_size=32,
                                        validation_data=(X_val, y_val),
                                        callbacks=[checkpoint_cb, early_stopping_cb])
        model_CNN_S.save(model_dir, save_format='tf')
        # save history for later plots - for loading history see test_cifar_tf_models.ipynb
        history_path = os.path.join(os.getcwd(), 'models', 'cifar', "history_" + model_name +".json")
        with open(history_path, "w") as f:
            json.dump(history_CNN_S.history, f)

    model_CNN_S.summary()

    y_hat_S = pd.DataFrame({'predicted': model_CNN_S.predict(X_test).argmax(1), 'true': y_test.argmax(axis=1)})
    y_hat_S['predicted_label'] = ic.label_prediction(y_hat_S['predicted'], label_names)
    y_hat_S['true_label'] = ic.label_prediction(y_hat_S['true'], label_names)

    test_loss_S, test_acc_S = model_CNN_S.evaluate(X_test, y_test)
    print('---' * 40)
    print('Test loss: {:.3f}'.format(test_loss_S))
    print('Test accuracy: {:.3f}'.format(test_acc_S))

    # ------------------------------------------------
    # MODEL CNN_M
    model_name = 'CNN_M'
    model_dir = os.path.join(cwd, 'models', 'cifar', 'model_' + model_name)
    # checkpoints to save weights
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weights_path, model_name, "weights.{epoch:02d}-{val_loss:.2f}.h5"),  # must be set for each model
        save_weights_only=True,
        save_best_only=True)
    print('---'*40)
    print('---' * 40)
    print(model_name + ':')
    # check if model exists:
    if os.path.exists(model_dir):
        print("The model exists. Loading...")
        model_CNN_M = keras.models.load_model(model_dir)  # load model
    else:
        print("The model does not exist. Creating and fitting a new model...")
        model_CNN_M = model_CNN_M(input_shape)

        model_CNN_M.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        history_CNN_M = model_CNN_M.fit(X_train, y_train, epochs=400, batch_size=32,
                                        validation_data=(X_val, y_val),
                                        callbacks=[checkpoint_cb, early_stopping_cb])
        model_CNN_M.save(model_dir, save_format='tf')
        # save history for later plots - for loading history see test_cifar_tf_models.ipynb
        history_path = os.path.join(os.getcwd(), 'models', 'cifar', "history_" + model_name +".json")
        with open(history_path, "w") as f:
            json.dump(history_CNN_M.history, f)

    model_CNN_M.summary()

    y_hat_M = pd.DataFrame({'predicted': model_CNN_M.predict(X_test).argmax(1), 'true': y_test.argmax(axis=1)})
    y_hat_M['predicted_label'] = ic.label_prediction(y_hat_M['predicted'], label_names)
    y_hat_M['true_label'] = ic.label_prediction(y_hat_M['true'], label_names)

    test_loss_M, test_acc_M = model_CNN_M.evaluate(X_test, y_test)
    print('---' * 40)
    print('Test loss: {:.3f}'.format(test_loss_M))
    print('Test accuracy: {:.3f}'.format(test_acc_M))

    # ------------------------------------------------
    # MODEL CNN_L
    model_name = 'CNN_L'
    model_dir = os.path.join(cwd, 'models', 'cifar', 'model_' + model_name)
    # checkpoints to save weights
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weights_path, model_name, "weights.{epoch:02d}-{val_loss:.2f}.h5"),  # must be set for each model
        save_weights_only=True,
        save_best_only=True)
    print('---'*40)
    print('---' * 40)
    print(model_name + ':')
    # check if model exists:
    if os.path.exists(model_dir):
        print("The model exists. Loading...")
        model_CNN_L = keras.models.load_model(model_dir)  # load model
    else:
        print("The model does not exist. Creating and fitting a new model...")
        model_CNN_L = model_CNN_L(input_shape)

        model_CNN_L.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        history_CNN_L = model_CNN_L.fit(X_train, y_train, epochs=400, batch_size=32,
                                        validation_data=(X_val, y_val),
                                        callbacks=[checkpoint_cb, early_stopping_cb])
        model_CNN_L.save(model_dir, save_format='tf')
        # save history for later plots - for loading history see test_cifar_tf_models.ipynb
        history_path = os.path.join(os.getcwd(), 'models', 'cifar', "history_" + model_name +".json")
        with open(history_path, "w") as f:
            json.dump(history_CNN_L.history, f)

    model_CNN_L.summary()

    y_hat_L = pd.DataFrame({'predicted': model_CNN_L.predict(X_test).argmax(1), 'true': y_test.argmax(axis=1)})
    y_hat_L['predicted_label'] = ic.label_prediction(y_hat_L['predicted'], label_names)
    y_hat_L['true_label'] = ic.label_prediction(y_hat_L['true'], label_names)

    test_loss_L, test_acc_L = model_CNN_L.evaluate(X_test, y_test)
    print('---' * 40)
    print('Test loss: {:.3f}'.format(test_loss_L))
    print('Test accuracy: {:.3f}'.format(test_acc_L))

    # ------------------------------------------------
    # MODEL CNN_XL
    model_name = 'CNN_XL'
    model_dir = os.path.join(cwd, 'models', 'cifar', 'model_' + model_name)
    # checkpoints to save weights
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weights_path, model_name, "weights.{epoch:02d}-{val_loss:.2f}.h5"),  # must be set for each model
        save_weights_only=True,
        save_best_only=True)
    print('---'*40)
    print('---' * 40)
    print(model_name + ':')
    # check if model exists:
    if os.path.exists(model_dir):
        print("The model exists. Loading...")
        model_CNN_XL = keras.models.load_model(model_dir)  # load model
    else:
        print("The model does not exist. Creating and fitting a new model...")
        model_CNN_XL = model_CNN_XL(input_shape)

        model_CNN_XL.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        history_CNN_XL = model_CNN_XL.fit(X_train, y_train, epochs=400, batch_size=32,
                                          validation_data=(X_val, y_val),
                                          callbacks=[checkpoint_cb, early_stopping_cb])
        model_CNN_XL.save(model_dir, save_format='tf')
        # save history for later plots - for loading history see test_cifar_tf_models.ipynb
        history_path = os.path.join(os.getcwd(), 'models', 'cifar', "history_" + model_name +".json")
        with open(history_path, "w") as f:
            json.dump(history_CNN_XL.history, f)

    model_CNN_XL.summary()

    y_hat_XL = pd.DataFrame({'predicted': model_CNN_XL.predict(X_test).argmax(1), 'true': y_test.argmax(axis=1)})
    y_hat_XL['predicted_label'] = ic.label_prediction(y_hat_XL['predicted'], label_names)
    y_hat_XL['true_label'] = ic.label_prediction(y_hat_XL['true'], label_names)

    test_loss_XL, test_acc_XL = model_CNN_XL.evaluate(X_test, y_test)
    print('---' * 40)
    print('Test loss: {:.3f}'.format(test_loss_XL))
    print('Test accuracy: {:.3f}'.format(test_acc_XL))


