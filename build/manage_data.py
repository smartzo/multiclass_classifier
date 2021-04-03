import pandas as pd
import numpy as np
import random
import os
import shutil
import pickle
import csv
from datetime import datetime
from config import TIME_COLUMN_NAME, TARGET_COLUMN_NAME
from config import DROP_COLUMN_THRESHOLD
from config import FIGURES_DIR_NAME
from config import LABEL2STATUS_PKL
from config import DROPPED_FEATURES_FILENAME
from config import TEST_DATA_SET_PERCENTAGE
from config import PATH_TO_INPUT_DATA
from config import TIME_SERIES_FILENAME
from config import TRAIN_DATA_SET_PERCENTAGE
from config import TEST_SET_FILENAME
from config import N_FOLDS

########################################################################################################################


class InputDataClass:
    def __init__(self, path_to_time_series):
        self.path_to_time_series = path_to_time_series
        self.matrix = None
        self.n_rows = 0
        self.column2nan = {}
        self.target_info = {}
        self.status2label = {}
        self.dropped_features = []
        self.n_classes = 0
        self.n_features = 0

        self.train_and_tune_size = 0
        self.train_and_tune_indices = None
        self.test_size = 0
        self.test_indices = None

    def load_data(self):

        self.matrix = pd.read_csv(
                                    self.path_to_time_series,
                                    encoding='utf-8',
                                    skiprows=0,
                                    sep=',',
                                    skipinitialspace=True,
                                    index_col=None,
                                    parse_dates=[TIME_COLUMN_NAME]
                                 )

        self.n_rows = len(self.matrix.index)

        return

    def remove_rows_with_nan_at_target_column(self):

        n_nan_rows = self.matrix[TARGET_COLUMN_NAME].isna().sum()

        if n_nan_rows / self.n_rows > DROP_COLUMN_THRESHOLD:
            quit("ABORT :: Target column contains too many NaN entries.")

        self.matrix.dropna(subset=[TARGET_COLUMN_NAME], inplace=True)

        self.n_rows -= n_nan_rows

        return

    def get_nan_entries_per_column(self):

        for column_name in self.matrix.columns:
            n_nan_rows = self.matrix[column_name].isna().sum()
            if n_nan_rows != 0:
                self.column2nan[column_name] = n_nan_rows

        temp = sorted(self.column2nan.items(), key=lambda x: x[1])
        for item in temp:
            print(item)

        return

    def get_target_info(self):

        value2indices = {}
        for idx in self.matrix.index:
            value = self.matrix.at[idx, TARGET_COLUMN_NAME]
            if value not in value2indices:
                value2indices[value] = []
            value2indices[value].append(idx)

        for label, value in enumerate(value2indices):
            frequency = len(value2indices[value])
            probability = frequency / self.n_rows
            indices = value2indices[value]
            ti = TargetInfoClass(value, label, frequency, probability, indices)
            self.target_info[label] = ti
            self.status2label[value] = label

        self.n_classes = len(value2indices)

        return

    def drop_features(self):

        for column_name, n_nan in self.column2nan.items():
            percentage = n_nan / self.n_rows
            if percentage > DROP_COLUMN_THRESHOLD:
                self.matrix.drop(column_name, axis=1, inplace=True)
                self.dropped_features.append(column_name)
                print("\nColumn", column_name, "with", n_nan, "NaN entries has been dropped.")
                # if column_name == TARGET_COLUMN_NAME:
                #     quit("ABORT :: Target column is dropped.")

        self.n_features = len(self.matrix.columns) - 2

        return

    def convert_timestamps_to_float(self):

        def timestamp2ole(timestamp_string):

            timestamp = datetime.strptime(timestamp_string, timestamp_format)

            delta = timestamp - OLE_TIME_ZERO

            return float(delta.days) + (float(delta.seconds) / 86400)

        OLE_TIME_ZERO = datetime(1899, 12, 30, 23, 59, 59)
        timestamp_format = '%Y-%m-%d %H:%M:%S'

        self.matrix[TIME_COLUMN_NAME] = self.matrix[TIME_COLUMN_NAME].map(lambda x: timestamp2ole(x))

        return

    def convert_categorical_values_to_integer(self):

        def categorical2integer(value):
            return self.status2label[value]

        self.matrix[TARGET_COLUMN_NAME] = self.matrix[TARGET_COLUMN_NAME].map(lambda x: categorical2integer(x))

        return

    def fill_missing_values(self):
        self.matrix.interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
        return

    def plot_time_series(self):
        import matplotlib.pyplot as plt

        if os.path.isdir(FIGURES_DIR_NAME):
            shutil.rmtree(FIGURES_DIR_NAME)

        os.mkdir(FIGURES_DIR_NAME)

        for column_name in self.column2nan:
            if column_name not in self.matrix.columns:
                continue
            fig = plt.figure(figsize=(10, 5))
            # plt.plot(self.matrix[column_name], '.')
            plt.plot(self.matrix[TARGET_COLUMN_NAME], self.matrix[column_name], '.')
            # plt.plot(self.matrix[TIME_COLUMN_NAME], self.matrix[column_name], '.')
            # plt.plot(self.matrix[TIME_COLUMN_NAME], '.-')
            plt.grid()
            figure_name = column_name + ".jpg"
            figure_path = os.path.join(FIGURES_DIR_NAME, figure_name)
            fig.savefig(figure_path)
            plt.clf()

    def get_label2datasize(self, percentage, all_data_size):

        label2datasize = {}

        datasize = int(percentage * all_data_size)

        my_sum = 0

        for label, info in self.target_info.items():

            probability = info.probability

            size = int(np.ceil(datasize * probability))

            label2datasize[label] = size

            my_sum += size

        datasize = my_sum

        return label2datasize, datasize

    @staticmethod
    def get_dataset_indices(label2datasize, index_pool):

        dataset_indices = {}

        for label, indices in index_pool.items():
            size = label2datasize[label]

            selected_indices = random.sample(indices, size)

            dataset_indices[label] = sorted(selected_indices)

        return dataset_indices

    def get_dataset_df(self, index_pool):

        dataframes = []

        for indices in index_pool.values():

            dataset_dataframe = self.matrix.loc[indices, :]

            dataframes.append(dataset_dataframe)

        dataset_dataframe = pd.concat(dataframes)

        return dataset_dataframe

    def set_dataset_attributes(self, train_and_tune_size, train_and_tune_indices, test_size, test_indices):

        self.train_and_tune_size = train_and_tune_size

        self.train_and_tune_indices = train_and_tune_indices

        self.test_size = test_size

        self.test_indices = test_indices

        return

    def dump_label2status(self):
        label2status = {}
        for label in self.target_info:
            value = self.target_info[label].value
            label2status[label] = value

        with open(LABEL2STATUS_PKL, 'wb') as output:
            pickle.dump(label2status, output, pickle.HIGHEST_PROTOCOL)

        return

    def write_dropped_features(self):
        with open(DROPPED_FEATURES_FILENAME, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(self.dropped_features)
        return

########################################################################################################################


class TargetInfoClass:
    def __init__(self, value, label, frequency, probability, indices):
        self.value = value
        self.label = label
        self.frequency = frequency
        self.probability = probability
        self.indices = indices

########################################################################################################################


class TrainTuneNumpyClass:

    def __init__(self, train_df, tune_df):
        self.train = self.FeaturesAndTargetClass(train_df)

        self.tune = self.FeaturesAndTargetClass(tune_df)

    class FeaturesAndTargetClass:
        """Store features & target"""

        def __init__(self, df):

            X, y = self.separate_features_and_target(df)

            self.features = np.array(X, dtype=float)

            self.target = np.array(y, dtype=float)

        @staticmethod
        def separate_features_and_target(df):

            X = df.drop(TARGET_COLUMN_NAME, axis=1)

            X = X.drop(TIME_COLUMN_NAME, axis=1)

            y = df[TARGET_COLUMN_NAME]

            return X, y

########################################################################################################################


def get_test(input_data):

    # ------------------------------------------------------------------------------------------------------------------

    label2testsize, test_size = input_data.get_label2datasize(TEST_DATA_SET_PERCENTAGE, input_data.n_rows)

    # ------------------------------------------------------------------------------------------------------------------

    test_index_pool = {k: v.indices for k, v in input_data.target_info.items()}

    test_indices = InputDataClass.get_dataset_indices(label2testsize, test_index_pool)

    # ------------------------------------------------------------------------------------------------------------------

    test_dataframe = input_data.get_dataset_df(test_indices)

    test_dataframe.to_csv(TEST_SET_FILENAME)

    # ------------------------------------------------------------------------------------------------------------------

    train_and_tune_size = input_data.n_rows - test_size

    train_and_tune_indices = {}

    for label, test_label_indices in test_indices.items():

        all_label_indices = input_data.target_info[label].indices

        remaining_indices = set(all_label_indices) - set(test_label_indices)

        train_and_tune_indices[label] = sorted(list(remaining_indices))

    # ------------------------------------------------------------------------------------------------------------------

    input_data.set_dataset_attributes(train_and_tune_size,
                                      train_and_tune_indices,
                                      test_size,
                                      test_indices)

    # ------------------------------------------------------------------------------------------------------------------

    return

########################################################################################################################


def get_train_and_tune(input_data, fold):

    # ------------------------------------------------------------------------------------------------------------------

    train_and_tune_size = input_data.train_and_tune_size

    train_and_tune_indices = input_data.train_and_tune_indices

    # ------------------------------------------------------------------------------------------------------------------

    label2trainsize, train_size = input_data.get_label2datasize(TRAIN_DATA_SET_PERCENTAGE, train_and_tune_size)

    train_indices = InputDataClass.get_dataset_indices(label2trainsize, train_and_tune_indices)

    # ------------------------------------------------------------------------------------------------------------------

    tune_size = train_and_tune_size - train_size

    tune_indices = {}

    for label, indices in train_and_tune_indices.items():
        tune_indices[label] = set(indices) - set(train_indices[label])

    # ------------------------------------------------------------------------------------------------------------------

    train_df = input_data.get_dataset_df(train_indices)

    tune_df = input_data.get_dataset_df(tune_indices)

    # ------------------------------------------------------------------------------------------------------------------

    ml_data = TrainTuneNumpyClass(train_df, tune_df)

    # ------------------------------------------------------------------------------------------------------------------

    info_string = "\n\nFold " + str(fold + 1) + "/" + str(N_FOLDS) + " :: "
    train_percentage = round(100 * train_size / train_and_tune_size, 1)
    info_string += " |Train| = " + str(train_size) + " (" + str(train_percentage) + "%),"
    tune_percentage = round(100 * tune_size / train_and_tune_size, 1)
    info_string += " |Tune| = " + str(tune_size) + " (" + str(tune_percentage) + "%)"
    print(info_string)

    # ------------------------------------------------------------------------------------------------------------------

    return ml_data

########################################################################################################################


def get_input_data():

    path_to_time_series = os.path.join(PATH_TO_INPUT_DATA, TIME_SERIES_FILENAME)

    input_data = InputDataClass(path_to_time_series)

    input_data.load_data()

    input_data.remove_rows_with_nan_at_target_column()

    input_data.get_nan_entries_per_column()

    input_data.drop_features()

    input_data.get_target_info()

    # input_data.convert_timestamps_to_float()

    input_data.convert_categorical_values_to_integer()

    input_data.fill_missing_values()

    # input_data.plot_time_series()

    input_data.dump_label2status()

    input_data.write_dropped_features()

    return input_data

########################################################################################################################
