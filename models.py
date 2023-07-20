import tensorflow as tf
from tensorflow import keras


def build_model_CNN1(dropout, dense_layers, num_filters):
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(num_filters, (3, 1), strides=2, activation='relu'))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(num_filters, (3, 1), strides=2, activation='relu'))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(num_filters, (2, 1), strides=2, activation='relu'))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(dense_layers, activation='relu'))
    model.add(keras.layers.Dropout(dropout))

    # output layer
    # model.add(keras.layers.Dense(1, activation='softmax'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model


def build_model_CNN2(dropout, dense_layers, num_filters):
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(num_filters, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(num_filters, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(num_filters, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(dense_layers, activation='relu'))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(dense_layers, activation='relu'))
    model.add(keras.layers.Dropout(dropout))

    # output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    #model.add(keras.layers.Dense(14, activation='softmax'))
    return model


def build_model_CNN3():
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(16, (3, 1), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 1)))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 4th conv layer
    model.add(keras.layers.Conv2D(32, (3, 1), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 1)))
    model.add(keras.layers.BatchNormalization())

    # 5th conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 6th conv layer
    model.add(keras.layers.Conv2D(64, (2, 1), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())

    # 7th conv layer
    model.add(keras.layers.Conv2D(128, (2, 1), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.AveragePooling2D((2, 2)))

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(2380, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    #model.add(keras.layers.Dense(14, activation='softmax'))
    return model


def build_model_RNN(dropout, dense_layers):
    model = keras.Sequential()

    model.add(tf.keras.layers.Masking(mask_value=0))

    model.add(keras.layers.LSTM(256, return_sequences=True))
    model.add(keras.layers.LSTM(256, return_sequences=True))
    model.add(keras.layers.LSTM(256))

    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model