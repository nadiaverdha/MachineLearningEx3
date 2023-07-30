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

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

import json
from keras.utils import plot_model
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

tf.random.set_seed(30)

if __name__ == '__main__':

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    valid_datagen = ImageDataGenerator(rescale=1./255)

    cwd = os.getcwd()
    train_dir = os.path.join(cwd, 'data', 'monkeys', 'training', 'training')
    valid_dir = os.path.join(cwd, 'data', 'monkeys', 'validation', 'validation')

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        seed=32,
                                                        shuffle=True,
                                                        class_mode='categorical')

    valid_generator = train_datagen.flow_from_directory(valid_dir,
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        seed=32,
                                                        shuffle=False,
                                                        class_mode='categorical')

    df_info = pd.read_csv(os.path.join(cwd, 'data', 'monkeys', 'monkey_labels.txt'))
    labels = list(df_info['Label'])
    class_names = list(df_info[' Common Name                   '])
    print('class names:', class_names)

    train_num = train_generator.samples
    valid_num = valid_generator.samples
    print('number of train samples:', train_num)
    print('number of test samples:', valid_num)
    print(df_info)

    # prepare checkpoints
    weights_path = os.path.join(cwd, 'models', 'monkeys', 'saved_weights')
    try:
        os.makedirs(weights_path)
        print("Directory 'saved_weights' created successfully")
    except OSError as error:
        print("Directory 'saved_weights' can not be created - it might exist already")

    # early stopping to avoid overfitting/save computational power -> test set is used for validation
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    # ------------------------------------------------
    # MODEL transfer learning
    model_name = 'VGG16_GAP1'
    model_dir = os.path.join(cwd, 'models', 'monkeys', 'model_' + model_name)
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
        base_model = VGG16(weights='imagenet', include_top=False,
                           input_shape=(224, 224, 3))  # exclude top layers and last global pooling laye
        avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
        # x = keras.layers.Dense(512, activation='relu')(x)
        output = keras.layers.Dense(10, activation='softmax')(avg)

        model = keras.Model(inputs=base_model.input, outputs=output)
        model.summary()

        for layer in base_model.layers:
            layer.trainable = False
        model.summary()

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()

        history = model.fit_generator(train_generator,
                                      steps_per_epoch=len(train_generator),
                                      epochs=50,
                                      validation_data=valid_generator,
                                      validation_steps=len(valid_generator),
                                      callbacks=[checkpoint_cb, early_stopping_cb],
                                      verbose=1)


        model.save(model_dir, save_format='tf')
        # save history for later plots - for loading history see test_cifar_tf_models.ipynb
        history_path = os.path.join(os.getcwd(), 'models', 'monkeys', "history_" + model_name +".json")
        with open(history_path, "w") as f:
            json.dump(history.history, f)
        # save history for later plots - for loading history see test_cifar_tf_models.ipynb


    test_loss, test_acc = model.evaluate(valid_generator)
    print('---' * 40)
    print('Test loss: {:.3f}'.format(test_loss))
    print('Test accuracy: {:.3f}'.format(test_acc))