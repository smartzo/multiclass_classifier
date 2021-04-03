import os
import shutil
import joblib
import pandas as pd

from config import OPTUNA_DUMP, TUNE, ALL_TRIALS, BEST_TRIAL
from config_pytorch import PYTORCH
from config_xgboost import XGBOOST
import optuna_xgboost
import optuna_pytorch

########################################################################################################################


class OptunaClass:
    def __init__(self, name, script):
        self.ml_algorithm_name = name
        self.ml_algorithm_script = script
        self.directory = None

########################################################################################################################


def make_optuna_dir():

    OPTUNA_MODELS = [OptunaClass(PYTORCH, optuna_pytorch),
                     OptunaClass(XGBOOST, optuna_xgboost)]

    working_dir = os.getcwd()

    optuna_dump_dir = os.path.join(working_dir, OPTUNA_DUMP)

    if os.path.isdir(optuna_dump_dir):
        shutil.rmtree(optuna_dump_dir)

    os.mkdir(optuna_dump_dir)

    for optuna_model in OPTUNA_MODELS:
        name = optuna_model.ml_algorithm_name
        model_dir = os.path.join(optuna_dump_dir, name)
        os.mkdir(model_dir)
        optuna_model.directory = model_dir

    return OPTUNA_MODELS

########################################################################################################################


def make_tune_dir(optuna_dir, voyage):
    tune_dir = os.path.join(optuna_dir, TUNE + str(voyage))
    if os.path.isdir(tune_dir):
        shutil.rmtree(tune_dir)
    os.mkdir(tune_dir)

    all_trials_dir = os.path.join(tune_dir, ALL_TRIALS)
    os.mkdir(all_trials_dir)

    best_trial_dir = os.path.join(tune_dir, BEST_TRIAL)
    os.mkdir(best_trial_dir)

    return all_trials_dir, best_trial_dir

########################################################################################################################


def dump_model(all_trials_dir, trial_number, model):
    model_path = os.path.join(all_trials_dir, str(trial_number) + '.joblib')
    joblib.dump(model, model_path)
    return

########################################################################################################################


def dump_parameters(all_trials_dir, study):
    df_results = study.trials_dataframe()
    path_to_filename = os.path.join(all_trials_dir, 'df_optuna_results.pkl')
    df_results.to_pickle(path_to_filename)
    path_to_filename = os.path.join(all_trials_dir, 'df_optuna_results.csv')
    df_results.to_csv(path_to_filename)
    return

########################################################################################################################


def copy_best_trial(all_trials_dir, best_trial_dir, best_trial_number):
    filename = str(best_trial_number) + '.joblib'
    src = os.path.join(all_trials_dir, filename)
    dst = os.path.join(best_trial_dir, filename)
    shutil.copyfile(src, dst)
    return

########################################################################################################################


def load_best_trial(all_trials_dir, best_trial_dir):
    path_to_filename = os.path.join(all_trials_dir, 'df_optuna_results.pkl')
    study_params = pd.read_pickle(path_to_filename)

    filename = os.listdir(best_trial_dir)[0]
    path_to_filename = os.path.join(best_trial_dir, filename)
    model = joblib.load(path_to_filename)

    trial_number = int(filename.split('.')[0])

    return model, trial_number, study_params

########################################################################################################################
