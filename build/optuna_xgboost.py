import numpy as np
import xgboost
import optuna
from sklearn.metrics import balanced_accuracy_score

from config_xgboost import EVAL_METRIC
from config_xgboost import N_WARMUP_STEPS, N_TRIALS, TIMEOUT
from config_xgboost import MAXIMUM_BOOSTING_ROUNDS, EARLY_STOPPING_ROUNDS
import manage_optuna

########################################################################################################################


class Objective(object):

    def __init__(self, X_train, y_train, X_tune, y_tune, all_trials_dir, n_classes):
        self.dtrain = xgboost.DMatrix(X_train, label=y_train)
        self.dtune = xgboost.DMatrix(X_tune, label=y_tune)
        self.y_tune = y_tune
        self.all_trials_dir = all_trials_dir
        self.n_classes = n_classes

    def __call__(self, trial):

        param = {
            "verbosity": 0,
            "objective": "multi:softmax",
            'num_class': self.n_classes,
            "eval_metric": EVAL_METRIC,
            "booster": "gbtree",
            "reg_lambda": trial.suggest_float("reg_lambda", np.exp(-9), np.exp(2), step=None, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", np.exp(-9), np.exp(2), step=None, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 9),
            "learning_rate": trial.suggest_float("learning_rate", np.exp(-7), 1.0, step=None, log=True),
            "gamma": trial.suggest_float("gamma", np.exp(-16), np.exp(2), step=None, log=True),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "colsample_bytree": trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1.0, 0.05),
            "colsample_bylevel": trial.suggest_discrete_uniform('colsample_bylevel', 0.5, 1.0, 0.05),
            "min_child_weight": trial.suggest_float("min_child_weight", np.exp(-16), np.exp(5), step=None, log=True),
            "subsample": trial.suggest_discrete_uniform('subsample', 0.5, 1.0, 0.1),
            "base_score": 0.5,
            "colsample_bynode": 1,
            "gpu_id": -1,
            "importance_type": 'gain',
            "interaction_constraints": '',
            "max_delta_step": 0,
            "missing": "nan",
            "monotone_constraints": '()',
            "n_jobs": 0,
            "num_parallel_tree": 1,
            "random_state": 0,
            "scale_pos_weight": 1,
            "tree_method": 'exact',
            "validate_parameters": 1
        }

        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation-' + EVAL_METRIC)

        xgb_model = xgboost.train(param,
                                  self.dtrain,
                                  evals=[(self.dtune, 'validation')],
                                  num_boost_round=MAXIMUM_BOOSTING_ROUNDS,
                                  early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                                  callbacks=[pruning_callback],
                                  verbose_eval=False)

        manage_optuna.dump_model(self.all_trials_dir, trial.number, xgb_model)

        # trial_mlogloss = xgb_model.best_score

        preds = xgb_model.predict(self.dtune)
        pred_labels = np.rint(preds)
        trial_accuracy = round(100 * balanced_accuracy_score(self.y_tune, pred_labels), 4)

        return trial_accuracy

########################################################################################################################


def execute(input_data, ml_np, optuna_dir, fold):

    # ------------------------------------------------------------------------------------------------------------------

    n_classes = input_data.n_classes

    # ------------------------------------------------------------------------------------------------------------------

    X_train = ml_np.train.features

    y_train = ml_np.train.target

    X_tune = ml_np.tune.features

    y_tune = ml_np.tune.target

    # ------------------------------------------------------------------------------------------------------------------

    all_trials_dir, best_trial_dir = manage_optuna.make_tune_dir(optuna_dir, fold)

    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=N_WARMUP_STEPS), direction="maximize")

    study.optimize(Objective(X_train, y_train, X_tune, y_tune, all_trials_dir, n_classes),
                   n_trials=N_TRIALS,
                   timeout=TIMEOUT)

    # ------------------------------------------------------------------------------------------------------------------

    manage_optuna.dump_parameters(all_trials_dir, study)

    best_trial = study.best_trial

    manage_optuna.copy_best_trial(all_trials_dir, best_trial_dir, best_trial.number)

    # ------------------------------------------------------------------------------------------------------------------

    return
