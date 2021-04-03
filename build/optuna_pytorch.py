import os
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from config import STANDARD_SCALER, FEATURES, TARGET
from config_pytorch import N_WARMUP_STEPS, N_TRIALS, TIMEOUT
from config_pytorch import SMALL_BATCHSIZE_NAME, MEDIUM_BATCHSIZE_NAME
from config_pytorch import BATCHSIZE_NAMES
import manage_optuna

########################################################################################################################

DEVICE = torch.device("cpu")

########################################################################################################################


class Objective(object):

    def __init__(self, X_train, y_train, X_tune, y_tune, all_trials_dir, n_classes, n_features, target_scaler):

        self.train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                            torch.from_numpy(y_train).type(torch.LongTensor))

        self.tune_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_tune).float(),
                                                           torch.from_numpy(y_tune).type(torch.LongTensor))

        self.all_trials_dir = all_trials_dir

        self.n_features = n_features

        self.n_classes = n_classes

        self.target_scaler = target_scaler

    def __call__(self, trial):

        # Generate the model.
        model = define_model(trial, self.n_features, self.n_classes).to(DEVICE)

        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
        # optimizer_name = "Adam"
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

        EPOCHS = trial.suggest_int("epochs", 10, 100, 5)

        batch_size_name = trial.suggest_categorical("batch_size", BATCHSIZE_NAMES)
        BATCHSIZE = select_batchsize(batch_size_name)

        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=BATCHSIZE, shuffle=True)
        tune_loader = torch.utils.data.DataLoader(self.tune_dataset, batch_size=BATCHSIZE, shuffle=True)

        N_TRAIN_EXAMPLES = BATCHSIZE * 30 * 3
        N_VALID_EXAMPLES = BATCHSIZE * 10 * 3
        trial_accuracy = 0

        # Training of the model.
        for epoch in range(EPOCHS):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # Limiting training data for faster epochs.
                if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                    break

                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

            # Validation of the model.
            model.eval()
            true = []
            preds = []
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(tune_loader):
                    # Limiting validation data.
                    if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                        break
                    data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                    output = model(data)
                    # Get the index of the max log-probability.
                    pred = output.argmax(dim=1, keepdim=True)
                    preds += pred.tolist()
                    true += target.tolist()

            trial_accuracy = round(100 * balanced_accuracy_score(true, preds), 4)

            trial.report(trial_accuracy, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        manage_optuna.dump_model(self.all_trials_dir, trial.number, model)

        return trial_accuracy

########################################################################################################################


def select_batchsize(batchsize_name):

    if batchsize_name == SMALL_BATCHSIZE_NAME:
        batchsize = 32
    elif batchsize_name == MEDIUM_BATCHSIZE_NAME:
        batchsize = 64
    else:
        batchsize = 128

    return batchsize

########################################################################################################################


def define_model(trial, in_features, n_classes):

    layers = []
    n_layers = trial.suggest_int("n_layers", 2, 3)

    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 5, 515, 5)

        layers.append(nn.Linear(in_features, out_features))

        layers.append(nn.ReLU())

        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features

    layers.append(nn.Linear(in_features, n_classes))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

########################################################################################################################


def apply_standard_scaling(D_train, D_tune, D_type, voyage):

    def values2string(values):
        value = round(values[0], 6)
        line = str(value)
        for i in range(1, len(values)):
            value = round(values[i], 6)
            line += ";" + str(value)
        line += "\n"

        return line

    # ------------------------------------------------------------------------------------------------------------------

    working_dir = os.getcwd()
    standard_scaler_dir = os.path.join(working_dir, STANDARD_SCALER)

    if not os.path.isdir(standard_scaler_dir):
        os.mkdir(standard_scaler_dir)

    # ------------------------------------------------------------------------------------------------------------------

    if D_type == TARGET:
        D_train = D_train.reshape(-1, 1)
        D_tune = D_tune.reshape(-1, 1)

    # ------------------------------------------------------------------------------------------------------------------

    scaler = StandardScaler()
    scaler.fit(D_train)

    # ------------------------------------------------------------------------------------------------------------------

    filename = str(voyage) + "."

    if D_type == FEATURES:
        filename += FEATURES
    elif D_type == TARGET:
        filename += TARGET
    else:
        raise Exception("Incorrect data set for StandardScaler")

    path_to_file = os.path.join(standard_scaler_dir, filename)

    mean_line = values2string(scaler.mean_)
    std_line = values2string(scaler.scale_)
    with open(path_to_file, "w") as f:
        f.write(mean_line)
        f.write(std_line)

    # ------------------------------------------------------------------------------------------------------------------

    D_train = scaler.transform(D_train)
    D_tune = scaler.transform(D_tune)

    if D_type == TARGET:
        n_train = len(D_train)
        D_train = D_train.reshape((n_train,))

        n_tune = len(D_tune)
        D_tune = D_tune.reshape((n_tune,))

    return D_train, D_tune, scaler

########################################################################################################################


def execute(input_data, ml_np, optuna_dir, fold):

    # ------------------------------------------------------------------------------------------------------------------

    n_classes = input_data.n_classes

    n_features = input_data.n_features

    # ------------------------------------------------------------------------------------------------------------------

    X_train = ml_np.train.features

    y_train = ml_np.train.target

    X_tune = ml_np.tune.features

    y_tune = ml_np.tune.target

    # ------------------------------------------------------------------------------------------------------------------

    X_train, X_tune, train_scaler = apply_standard_scaling(X_train, X_tune, FEATURES, fold)

    # ------------------------------------------------------------------------------------------------------------------

    all_trials_dir, best_trial_dir = manage_optuna.make_tune_dir(optuna_dir, fold)

    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=N_WARMUP_STEPS), direction="maximize")

    study.optimize(Objective(X_train, y_train, X_tune, y_tune, all_trials_dir, n_classes, n_features, None),
                   n_trials=N_TRIALS,
                   timeout=TIMEOUT)

    # ------------------------------------------------------------------------------------------------------------------

    manage_optuna.dump_parameters(all_trials_dir, study)

    best_trial = study.best_trial

    manage_optuna.copy_best_trial(all_trials_dir, best_trial_dir, best_trial.number)

    # ------------------------------------------------------------------------------------------------------------------

    return

########################################################################################################################
