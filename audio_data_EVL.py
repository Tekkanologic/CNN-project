import json
import pandas as pd
import numpy as np
import math
import tensorflow as tf

'Definitions'
TRAIN_PATH = r'E:/DATA/preproc/13MFCCsLabelledSpeakers/train_data.json'
DEV_PATH = r'E:/DATA/preproc/13MFCCsLabelledSpeakers/dev_data.json'
EVAL_PATH = r'E:/DATA/preproc/13MFCCsLabelledSpeakers/eval_data.json'
bonaOnly = ['0049',
            '0050',
            '0051',
            '0052',
            '0053',
            '0054',
            '0055',
            '0056',
            '0057',
            '0058',
            '0059',
            '0060',
            '0061',
            '0062',
            '0063',
            '0064',
            '0065',
            '0066',
            '0067',
            '0068',
            '0099',
            '0100',
            '0101',
            '0102',
            '0103',
            '0104',
            '0105',
            '0106',
            '0107',
            '0108'
            ]


class PrepData:

    def __init__(self, valid_size, test_size, batch_size):
        self.valid_size = valid_size
        self.test_size = test_size
        self.batch_size = batch_size

    @staticmethod
    def find_max_list(list):
        list_len = [len(i) for i in list]
        return max(list_len)

    @staticmethod
    def load_data(PATH):
        print(PATH[40:], " - Data loading...")
        with open(PATH, 'r') as file:
            data = json.load(file)
        print(PATH[40:], " - Data loaded!")
        return data

    @staticmethod
    def create_datasets(df, speaker_data):
        df_2 = pd.DataFrame()
        for speaker in speaker_data:
            reg_df = df.loc[df["id"] == speaker]
            for classname in reg_df["class"].unique():
                A_df = reg_df.loc[reg_df["class"] == classname]
                df_2 = df_2.append(A_df)
            # move back into for loop for duplicate bonafide classes
            bon_df = reg_df.loc[reg_df["class"] == 'bonafide']
            df_2 = df_2.append(bon_df)

        print("value counts:", df_2.label.value_counts())
        return df_2

    def prep_data(self):
        'load data and combine together into dataframe'

        evl = self.load_data(EVAL_PATH)
        df = pd.DataFrame(evl)
        del evl

        'Create train, test and validation datasets containing just the A10 spoof and bonafide from each speaker ID'

        print("Data prepping...")

        IDx = df.id.unique()  # gets array of speaker ids
        IDx = np.setdiff1d(IDx, bonaOnly, assume_unique=True)  # removes certain speaker IDs (bonafide only)

        size = len(IDx)
        train = IDx

        valid = np.random.choice(train, int(math.floor(size * self.valid_size)), replace=True)
        train = np.setdiff1d(train, valid, assume_unique=True)
        test = np.random.choice(train, int(math.floor(size * self.test_size)), replace=True)
        train = np.setdiff1d(train, test, assume_unique=True)

        # creates dataframes containing speakers pairs with equal number of spoof/bonafide
        print("---------------------")
        print("Train", np.sort(train))
        train_df = self.create_datasets(df, train)
        print("Valid", np.sort(valid))
        test_df = self.create_datasets(df, test)
        print("Test", np.sort(test))
        valid_df = self.create_datasets(df, valid)
        print("---------------------")
        print("1/6 Dataframes loaded!")

        # get the maximum length of the array
        list_df = pd.DataFrame()
        list_df = list_df.append(train_df)
        list_df = list_df.append(test_df)
        list_df = list_df.append(valid_df)
        length_x = np.array(list_df.mfcc)
        maxlen = self.find_max_list(length_x)
        del length_x, list_df
        print("2/6 Max length calculated!")

        # create train, test and validation splits
        X_train = np.array(train_df.mfcc)
        y_train = np.array(train_df.label)
        del train_df
        X_valid = np.array(valid_df.mfcc)
        y_valid = np.array(valid_df.label)
        del valid_df
        X_test = np.array(test_df.mfcc)
        y_test = np.array(test_df.label)
        del test_df
        print("3/6 Train, test split")

        # pads dataset with zeros up to maxlength calculated step 2/6
        X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=maxlen)
        print("4/6 - 1/3 Zero padding")
        X_valid = tf.keras.preprocessing.sequence.pad_sequences(X_valid, padding='post', maxlen=maxlen)
        print("4/6 - 2/3 Zero padding")
        X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', maxlen=maxlen)
        print("4/6 - 3/3 Zero padding")

        # creates extra dimension, required for CNN
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        X_valid = X_valid[..., np.newaxis]
        print("5/6 adding extra dimension")

        # create tensorflow dataset objects
        train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train = train.shuffle(X_train.size).batch(batch_size=self.batch_size, drop_remainder=True)
        del X_train, y_train
        print("6/6 - 1/3 Creating train tensor")

        test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test = test.shuffle(X_test.size).batch(batch_size=self.batch_size, drop_remainder=True)
        del X_test, y_test
        print("6/6 - 2/3 Creating test tensor")

        valid = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
        valid = valid.shuffle(X_valid.size).batch(batch_size=self.batch_size, drop_remainder=True)
        del X_valid, y_valid
        print("6/6 - 3/3 Creating valid tensor")

        print("6/6 Created tensors!")
        print("All data prep tasks completed!")

        return train, test, valid, maxlen