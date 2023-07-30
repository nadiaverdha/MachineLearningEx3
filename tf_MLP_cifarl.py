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

class MLPModel(keras.Model):
    def __init__(self, neurons_per_layer, num_classes=10):
        super(MLPModel, self).__init__()
        self.flatten = keras.layers.Flatten()
        self.hidden_layers = [tf.keras.layers.Dense(neurons_per_layer[_], activation='relu')
                              for _ in range(len(neurons_per_layer))]
        self.output_ = keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.output_(x)
        return x


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, label_names = ic.load_train_test_data()
    # show_sample(label_names=label_names)
    y_train, y_test = ic.encode_y(y_train, y_test)  # perform OneHotEncoding
    # show_sample(label_names=label_names, sample_nr=5000)

    X_train = X_train.astype('float32') / 255
    X_test = X_test / 255.

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=30)

    # prepare checkpoints
    # early stopping to avoid overfitting/save computational power -> test set is used for validation
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    # checkpoints to save weights
    cwd = os.getcwd()
    weights_path = os.path.join(cwd, 'models', 'cifar', 'saved_weights')
    try:
        os.makedirs(weights_path)
        print("Directory 'saved_weights' created successfully")
    except OSError as error:
        print("Directory 'saved_weights' can not be created - it might exist already")

    # ------------------------------------------------
    # MODEL S
    print('Model S:')
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weights_path, 'Model_S', "weights.{epoch:02d}-{val_loss:.2f}.h5"),  # must be set for each model
        save_weights_only=True,
        save_best_only=True)

    # check if model exists:
    model_name = 'model_MLP_S'
    model_dir = os.path.join(cwd, 'models', 'cifar', model_name)
    if os.path.exists(model_dir):
        print("The model exists. Loading...")
        model_S = keras.models.load_model(model_dir)  # load model
    else:
        print("The model does not exist. Creating and fitting a new model...")
        model_S = MLPModel(neurons_per_layer=[500, 500, 500], num_classes=10)
        model_S.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        history_S = model_S.fit(X_train, y_train, epochs=400, batch_size=32,
                                validation_data=(X_val, y_val),
                                callbacks=[checkpoint_cb, early_stopping_cb])
        model_S.save(model_dir, save_format='tf')
        # save history for later plots - for loading history see test_cifar_tf_models.ipynb
        history_path = os.path.join(os.getcwd(), 'models', 'cifar', "history_MLP_S.json")
        with open(history_path, "w") as f:
            json.dump(history_S.history, f)
    model_S.summary()

    y_hat_S = pd.DataFrame({'predicted': model_S.predict(X_test).argmax(1),
                            'true': y_test.argmax(axis=1)})
    y_hat_S['predicted_label'] = ic.label_prediction(y_hat_S['predicted'], label_names)
    y_hat_S['true_label'] = ic.label_prediction(y_hat_S['true'], label_names)

    test_loss_S, test_acc_S = model_S.evaluate(X_test, y_test)
    print('---' * 40)
    print('Test loss: {:.3f}'.format(test_loss_S))
    print('Test accuracy: {:.3f}'.format(test_acc_S))

    # ------------------------------------------------
    # MODEL M
    print('Model M:')
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weights_path, 'Model_M', "weights.{epoch:02d}-{val_loss:.2f}.h5"),  # must be set for each model
        save_weights_only=True,
        save_best_only=True)

    # check if model exists:
    model_name = 'model_MLP_M'
    model_dir = os.path.join(cwd, 'models', 'cifar', model_name)
    if os.path.exists(model_dir):
        print("The model exists. Loading...")
        model_M = keras.models.load_model(model_dir)  # load model
    else:
        print("The model does not exist. Creating and fitting a new model...")
        model_M = MLPModel(neurons_per_layer=[1000, 1000, 1000], num_classes=10)
        model_M.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        history_M = model_M.fit(X_train, y_train, epochs=400, batch_size=32,
                                validation_data=(X_val, y_val),
                                callbacks=[checkpoint_cb, early_stopping_cb])
        model_M.save(model_dir, save_format='tf')
        # save history for later plots - for loading history see test_cifar_tf_models.ipynb
        history_path = os.path.join(os.getcwd(), 'models', 'cifar', "history_MLP_M.json")
        with open(history_path, "w") as f:
            json.dump(history_M.history, f)

    model_M.summary()

    y_hat_M = pd.DataFrame({'predicted': model_M.predict(X_test).argmax(1),
                            'true': y_test.argmax(axis=1)})
    y_hat_M['predicted_label'] = ic.label_prediction(y_hat_M['predicted'], label_names)
    y_hat_M['true_label'] = ic.label_prediction(y_hat_M['true'], label_names)

    test_loss_M, test_acc_M = model_M.evaluate(X_test, y_test)
    print('---' * 40)
    print('Test loss: {:.3f}'.format(test_loss_M))
    print('Test accuracy: {:.3f}'.format(test_acc_M))

    # ------------------------------------------------
    # MODEL L
    print('Model L:')
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weights_path, 'Model_L', "weights.{epoch:02d}-{val_loss:.2f}.h5"),  # must be set for each model
        save_weights_only=True,
        save_best_only=True)

    # check if model exists:
    model_name = 'model_MLP_L'
    model_dir = os.path.join(cwd, 'models', 'cifar', model_name)
    if os.path.exists(model_dir):
        print("The model exists. Loading...")
        model_L = keras.models.load_model(model_dir)  # load model
    else:
        print("The model does not exist. Creating and fitting a new model...")
        model_L = MLPModel(neurons_per_layer=[3000, 3000, 1000], num_classes=10)
        model_L.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        history_L = model_L.fit(X_train, y_train, epochs=400, batch_size=32,
                                validation_data=(X_val, y_val),
                                callbacks=[checkpoint_cb, early_stopping_cb])
        model_L.save(model_dir, save_format='tf')
        # save history for later plots - for loading history see test_cifar_tf_models.ipynb
        history_path = os.path.join(os.getcwd(), 'models', 'cifar', "history_MLP_L.json")
        with open(history_path, "w") as f:
            json.dump(history_L.history, f)
    model_L.summary()

    y_hat_L = pd.DataFrame({'predicted': model_L.predict(X_test).argmax(1),
                            'true': y_test.argmax(axis=1)})
    y_hat_L['predicted_label'] = ic.label_prediction(y_hat_L['predicted'], label_names)
    y_hat_L['true_label'] = ic.label_prediction(y_hat_L['true'], label_names)

    test_loss_L, test_acc_L = model_L.evaluate(X_test, y_test)
    print('---' * 40)
    print('Test loss: {:.3f}'.format(test_loss_L))
    print('Test accuracy: {:.3f}'.format(test_acc_L))

