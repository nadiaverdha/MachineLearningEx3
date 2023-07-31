"""
File for running all models
"""
import import_cifar as ic
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
import json
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical


def run_model(model_name, dataset, summary=False,
              plot_curves=False, save_curves=False, conf_matrix=False, save_conf_matrix=False,
              save_path=None):
    """
    :param model_name: name of the model
    :param dataset: Str: 'cifar' or 'monkeys'
    :param summary: True/False(default) - show the model summary
    :param plot_curves: True/False(default) - show loss/acc curves
    :param save_curves: True/False(default) - save loss/acc curves
    :param conf_matrix: True/False(default) - plot the confusion matrix
    :param save_confusion_matrix: True/False(default) - save te confusion matrix
    :param save_path: path to folder for saving outputs (default=None)
    :return:
    """
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, 'models', dataset, 'model_' + model_name)
    model = keras.models.load_model(model_dir)  # load model
    print(model_name, ':')
    if dataset == 'cifar':
        # load data
        _, _, X_test, y_test, label_names = ic.load_train_test_data()
        X_test = X_test / 255.
        if model_name == 'transfer_learning':
            X_test = tf.image.resize(X_test, (192, 192))
        # evaluate
        y_test = tf.one_hot(y_test, 10)
        test_loss, test_acc = model.evaluate(X_test, y_test)
        # y_hat = pd.DataFrame({'predicted': model.predict(X_test).argmax(1),
        #                       'true': y_test.argmax(axis=1)})
        # y_hat['predicted_label'] = ic.label_prediction(y_hat['predicted'], label_names)
        # y_hat['true_label'] = ic.label_prediction(y_hat['true'], label_names)
        y_test = y_test.numpy().argmax(1)
    elif dataset == 'monkeys':
        test_dir = os.path.join(cwd, 'data', 'monkeys', 'validation', 'validation')
        test_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(test_dir,
                                                                               target_size=(224, 224),
                                                                               shuffle=False)
        X_test = test_data_gen
        y_test = test_data_gen.classes
        df_info = pd.read_csv(os.path.join(cwd, 'data', 'monkeys', 'monkey_labels.txt'))

        test_loss, test_acc = model.evaluate(X_test)

        # get label_names
        label_names_raw = list(df_info[' Common Name                   '].values)
        label_names = []
        for raw in label_names_raw:
            label = raw.replace(' ', '')
            label = label.replace('_', ' ')
            label_names.append(label)

    else:
        print("dataset_name must be 'cifar' or 'monkeys'")
        return

    print('Test loss: {:.3f}'.format(test_loss))
    print('Test accuracy: {:.3f}'.format(test_acc))

    if summary:
        print(model.summary())

    if plot_curves:
        history_path = os.path.join(os.getcwd(), 'models', dataset)
        with open(os.path.join(history_path, 'history_' + model_name + ".json"), "r") as f:
            loaded_history = json.load(f)
        plt.figure()  #figsize=(12, 8)
        plt.subplot(1, 2, 1)
        plt.plot(loaded_history['accuracy'], label='training accuracy')
        plt.plot(loaded_history['val_accuracy'], label='validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(loaded_history['loss'], label='training loss')
        plt.plot(loaded_history['val_loss'], label='validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()  # Increase the value if more spacing is needed
        plt.subplots_adjust(wspace=0.5)
        plt.tight_layout()
        if save_curves:  # when it is shown the image can´t be saved -> save before
            plt.savefig(save_path + 'loss_curve_' + dataset + '_' + model_name + '.png', dpi=300)
            print(f'figure {"loss_curve_" + dataset + "_" + model_name + ".png"} saved to:', save_path)
        plt.show()

    if conf_matrix:
        y_hat = pd.DataFrame({'predicted': model.predict(X_test).argmax(1),
                              'true': y_test})
        conf_mat = confusion_matrix(y_hat.true, y_hat.predicted)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        if save_conf_matrix:  # when it is shown the image can´t be saved -> save before
            plt.savefig(save_path + 'cm_' + dataset + '_' + model_name + '.png', dpi=300)
            print(f'figure {"cm_" + dataset + "_" + model_name + ".png"} saved to:', save_path)
        plt.show()

    print('----' * 40)
    print('----' * 40)






if __name__ == '__main__':

    # evaluate monkeys models
    model_names = ['inception_MLP1', 'inception_MLP2', 'inception_GAP1', 'inception_GAP2', 'VGG16_GAP1', 'CNN_L', 'CNN_L_aug']
    for model_name in model_names:
        run_model(model_name, 'monkeys', summary=False,
                  plot_curves=False, save_curves=True, conf_matrix=True, save_conf_matrix=False,
                  save_path=r"C:/Users/ernstmar/Pictures/ML3/")

    # evaluate cifar models
    model_names = ['MLP_S', 'MLP_M', 'MLP_L', 'CNN_S', 'CNN_M', 'CNN_L']
    for model_name in model_names:
        run_model(model_name, 'cifar', summary=False,
                  plot_curves=True, save_curves=False, conf_matrix=True, save_conf_matrix=False,
                  save_path=r"C:/Users/ernstmar/Pictures/ML3/")


    model_name = 'transfer_learning'  # needs about 5 min
    run_model(model_name, 'cifar', summary=True,
              plot_curves=True, save_curves=False, conf_matrix=True, save_conf_matrix=False,
              save_path=r"C:/Users/ernstmar/Pictures/ML3/")