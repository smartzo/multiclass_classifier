
import os
import shutil
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, balanced_accuracy_score, accuracy_score

from config import PATH_TO_OPTUNA_DUMP, PATH_TO_TEST_SET, PATH_TO_STANDARD_SCALER
from config import TRAINSET, TUNESET, TESTSET
from config import PREDICTIONS, BEST_TRIAL
from config import FEATURES

########################################################################################################################


class ExperimentClass:

    def __init__(self, path_to_experiment):
        self.best_optuna_trial_path = os.path.join(path_to_experiment, BEST_TRIAL)
        self.eval_on_dataset = EvaluateOnDatasetClass()

    def load_predictions(self, data_set_type):
        if data_set_type == TESTSET:
            eval_manager = self.eval_on_dataset.test_set
            predictions_path = eval_manager.predictions_path
            y_pred = np.loadtxt(predictions_path)
            bacc_regr = eval_manager.scores.BACC.regr
            accr_regr = eval_manager.scores.ACCR.regr
        else:
            y_pred = None
            bacc_regr = None
            accr_regr = None

        return y_pred, [bacc_regr, accr_regr]

########################################################################################################################


class EvaluateOnDatasetClass:

    class ResultsClass:
        def __init__(self):
            self.scores = MetricsClass()
            self.predictions_path = None

        def dump_predictions(self, dataset_type, model_name, experiment_num, y_pred):
            predictions_dir_path = _form_predictions_dir_path()
            filename = str(experiment_num) + ".txt"
            filename_path = os.path.join(predictions_dir_path, dataset_type, model_name, filename)
            np.savetxt(filename_path, y_pred)
            self.predictions_path = filename_path
            return

    def __init__(self):
        self.train_set = self.ResultsClass()
        self.tune_set = self.ResultsClass()
        self.test_set = self.ResultsClass()

########################################################################################################################


class MetricsClass:

    class ProblemClass:
        def __init__(self):
            self.clas = None
            self.regr = None

    def __init__(self):
        self.RMSE = self.ProblemClass()
        self.MAPE = self.ProblemClass()
        self.ACCR = self.ProblemClass()
        self.BACC = self.ProblemClass()

    def assign(self, y_true, y_pred):

        self.RMSE.regr = np.sqrt(mean_squared_error(y_true, y_pred))
        self.MAPE.regr = self.mean_absolute_percentage_error(y_true, y_pred)
        self.ACCR.regr = accuracy_score(y_true, y_pred) * 100
        self.BACC.regr = balanced_accuracy_score(y_true, y_pred) * 100

        y_pred_clas = np.round(y_pred)

        self.RMSE.clas = np.sqrt(mean_squared_error(y_true, y_pred_clas))
        self.MAPE.clas = self.mean_absolute_percentage_error(y_true, y_pred_clas)
        self.ACCR.clas = accuracy_score(y_true, y_pred_clas) * 100
        self.BACC.clas = balanced_accuracy_score(y_true, y_pred) * 100

        return

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred) / y_true) * 100

########################################################################################################################


class ScaleStatisticsClass:
    def __init__(self):
        self.mean = None
        self.std = None

########################################################################################################################


def _get_models_info():
    model_names = os.listdir(PATH_TO_OPTUNA_DUMP)

    models = {}

    for model_name in model_names:
        models[model_name] = {}
        path_to_model_name = os.path.join(PATH_TO_OPTUNA_DUMP, model_name)
        experiments = os.listdir(path_to_model_name)
        for experiment in experiments:
            path_to_experiment = os.path.join(path_to_model_name, experiment)
            _, experiment_num = experiment.split('_')
            experiment_num = int(experiment_num)
            models[model_name][experiment_num] = ExperimentClass(path_to_experiment)

    return models

########################################################################################################################


def _create_prediction_directories(model_names):

    predictions_dir_path = _form_predictions_dir_path()

    if os.path.isdir(predictions_dir_path):
        shutil.rmtree(predictions_dir_path)

    os.mkdir(predictions_dir_path)

    dataset_eval_dirs = [TRAINSET, TUNESET, TESTSET]

    for dataset_eval_dir in dataset_eval_dirs:

        dataset_eval_dir_path = os.path.join(predictions_dir_path, dataset_eval_dir)

        os.mkdir(dataset_eval_dir_path)

        for model_name in model_names:

            model_dir_path = os.path.join(dataset_eval_dir_path, model_name)

            os.mkdir(model_dir_path)

    return

########################################################################################################################


def _form_predictions_dir_path():

    working_dir_path = os.getcwd()

    predictions_dir_path = os.path.join(working_dir_path, PREDICTIONS)

    return predictions_dir_path

########################################################################################################################


def _load_test_set():

    test_df = pd.read_csv(PATH_TO_TEST_SET, sep=',', index_col=0)

    return test_df

########################################################################################################################


def _load_standard_scaler():

    standard_scaler = {}

    filenames = os.listdir(PATH_TO_STANDARD_SCALER)

    for filename in filenames:
        experiment_num, data_type = filename.split('.')
        experiment_num = int(experiment_num)

        if experiment_num not in standard_scaler:
            standard_scaler[experiment_num] = {}

        experiment_standard_scaler = standard_scaler[experiment_num]

        experiment_standard_scaler[data_type] = ScaleStatisticsClass()

        filename_path = os.path.join(PATH_TO_STANDARD_SCALER, filename)

        statistics = np.loadtxt(filename_path, delimiter=";")

        if data_type == FEATURES:
            experiment_standard_scaler[data_type].mean = statistics[0, :]
            experiment_standard_scaler[data_type].std = statistics[1, :]
        else:
            experiment_standard_scaler[data_type].mean = statistics[0]
            experiment_standard_scaler[data_type].std = statistics[1]

    return standard_scaler

########################################################################################################################


def load_model(experiment):

    best_optuna_trial_path = experiment.best_optuna_trial_path

    filename = os.listdir(best_optuna_trial_path)[0]

    filename_path = os.path.join(best_optuna_trial_path, filename)

    model = joblib.load(filename_path)

    return model

########################################################################################################################


def execute():

    models_info = _get_models_info()

    model_names = list(models_info.keys())

    _create_prediction_directories(model_names)

    test_set_df = _load_test_set()

    standard_scaler = _load_standard_scaler()

    return models_info, test_set_df, standard_scaler

########################################################################################################################
