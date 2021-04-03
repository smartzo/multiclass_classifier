import os
import numpy as np
import shutil
from itertools import chain, combinations
from sklearn.metrics import balanced_accuracy_score, accuracy_score
# from statistics import mode
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config import BACC_THRESHOLDS
from config import MODEL_DELIMITER
from config import BEST_COMBINATION
from config import TESTSET
from config import BEST_PER_ALGORITHM
from config import ALL_COMBINATIONS
from config import COMBINED_ALGORITHMS
from config import AGGREGATED_PREDICTIONS
from config import TARGET_COLUMN_NAME

#######################################################################################################################


class AggregatedPredictionsClass:

    def __init__(self):
        self.avrg_all_preds = None
        self.avrg_thr_preds = None
        self.best_all_preds = None
        self.best_thr_preds = None
        self.n_all = 0
        self.n_best_thr = 0

        self.best_models = {}
        self.avrg_all_scores = None
        self.avrg_thr_scores = None
        self.best_all_scores = None
        self.best_thr_scores = None

    def aggregate(self, y_pred, scores, model_nums, model_name, isCombinedAlgorithms):

        if self.best_all_scores is None:
            self.best_all_scores = scores
            self.avrg_all_preds = []
            self.best_all_preds = np.zeros_like(y_pred)

        self.avrg_all_preds.append(y_pred)
        self.n_all += 1

        if scores[0] >= self.best_all_scores[0]:
            self.best_all_preds = y_pred
            self.best_all_scores = scores
        isScoreAboveThreshold = (scores[0] > BACC_THRESHOLDS[model_name])

        if isScoreAboveThreshold or isCombinedAlgorithms:

            if model_name not in self.best_models:
                self.best_models[model_name] = set()
            for model_num in model_nums:
                self.best_models[model_name].add(model_num)

            if not isCombinedAlgorithms:

                if self.avrg_thr_preds is None:
                    self.avrg_thr_preds = []
                    self.best_thr_preds = np.zeros_like(y_pred)
                self.avrg_thr_preds.append(y_pred)

                self.n_best_thr += 1

                if scores[0] >= self.best_all_scores[0]:
                    self.best_thr_preds = y_pred

        return

    def majority_voting(self):

        def generic_vote(stacked_lists):

            vote = np.zeros(n_datapoints)

            n_stacks = len(stacked_lists)

            local_vote = np.zeros(n_stacks)

            for i in range(n_datapoints):
                for k in range(n_stacks):
                    local_vote[k] = stacked_lists[k][i]
                # vote[i] = mode(local_vote)
                vote[i] = int(stats.mode(local_vote)[0])

            return vote

        n_datapoints = len(self.avrg_all_preds[0])

        self.avrg_all_preds = generic_vote(self.avrg_all_preds)

        if self.n_best_thr != 0:
            self.avrg_thr_preds = generic_vote(self.avrg_thr_preds)

        return

    def assign_scores(self, y_true):

        def generic_assign(y, y_hat):

            y_hat = np.rint(y_hat)

            bacc = balanced_accuracy_score(y, y_hat) * 100

            accr = accuracy_score(y, y_hat) * 100

            return bacc, accr

        self.avrg_all_scores = generic_assign(y_true, self.avrg_all_preds)

        self.best_all_scores = generic_assign(y_true, self.best_all_preds)

        if self.n_best_thr != 0:

            self.avrg_thr_scores = generic_assign(y_true, self.avrg_thr_preds)

            self.best_thr_scores = generic_assign(y_true, self.best_thr_preds)

        return

#######################################################################################################################


def _get_y_true(data_frame):
    target = data_frame[TARGET_COLUMN_NAME]
    y = np.array(target, dtype=int)
    return y

#######################################################################################################################


def _powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

########################################################################################################################


def _plot_results(y_true, y_pred, scores, model_combination):

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    predict_color = 'r'
    actual_color = 'k'

    sorted_indices = np.argsort(y_true)

    y_true_sorted = [y_true[x] for x in sorted_indices]

    y_pred = np.rint(y_pred)

    y_pred_sorted = [y_pred[x] for x in sorted_indices]

    ax.plot(y_true_sorted, linestyle='', linewidth=0.1, marker='.', markersize=25, color=actual_color, label='Actual')

    ax.plot(y_pred_sorted, linestyle='', linewidth=0.1, marker='.', markersize=10, color=predict_color, label='Predicted')

    my_filename = ""
    n_models = len(model_combination)
    for i, model_name in enumerate(model_combination):
        if i == n_models - 1:
            my_filename += str(model_name)
        else:
            my_filename += str(model_name) + " + "

    my_title = my_filename + " :: "
    my_title += "BACC = " + str(round(scores[0], 1)) + "%, "
    my_title += "ACCR = " + str(round(scores[1], 1)) + "%"

    ax.grid(True)
    plt.xlabel("i", fontsize=20)
    plt.ylabel("y[i] (sorted)", fontsize=20)
    plt.legend(fontsize=14)
    plt.title(my_title, fontsize=20)

    fig.tight_layout()
    plt.show(block=False)
    fig.savefig(str(my_filename) + '.jpg')

    return


def _plot_confusion_matrix(y_true, y_pred, model_combination):

    classes = set()
    for v in y_true:
        classes.add(v)
    n_classes = len(classes)

    lines = np.arange(-0.5, n_classes + 0.5, step=1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    linewidth = 1.0

    for line in lines:
        ax.axvline(line, color='k', lw=linewidth, alpha=0.9)
        ax.axhline(line, color='k', lw=linewidth, alpha=0.9)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    true2pred = {}
    for c in classes:
        true2pred[c] = {}
        for _c in classes:
            true2pred[c][_c] = 0

    for c_true, c_pred in zip(y_true, y_pred):
        true2pred[c_true][c_pred] += 1

    for c_true in true2pred:
        n = 0
        for c_pred in true2pred[c_true]:
            n += true2pred[c_true][c_pred]

        for c_pred in true2pred[c_true]:
            p = true2pred[c_true][c_pred] / n

            x = np.linspace(c_pred - 0.5, c_pred + 0.5, num=100)

            ax.fill_between(x, c_true - 0.5, c_true + 0.5, facecolor=cm.Blues(p))
            ax.annotate(str(round(p, 4)), xy=(c_pred, c_true), fontsize=14, weight='bold')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    xvalues = np.round(np.arange(0, n_classes, step=1))
    xlabels = [str(x) for x in xvalues]

    plt.xticks(xvalues, xlabels, rotation=0)

    yvalues = np.round(np.arange(0, n_classes, step=1))
    ylabels = [str(x) for x in yvalues]

    plt.yticks(yvalues, ylabels)

    my_filename = ""
    n_models = len(model_combination)
    for i, model_name in enumerate(model_combination):
        if i == n_models - 1:
            my_filename += str(model_name)
        else:
            my_filename += str(model_name) + " + "
    plt.title(my_filename, fontsize=20)

    plt.xlabel("Predicted Label", fontsize=20)
    plt.ylabel("True Label", fontsize=20)
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.tick_params(axis='y', which='major', labelsize=14)
    fig.tight_layout()
    plt.show(block=False)
    # plt.show()

    my_filename += '_cm.jpg'
    fig.savefig(my_filename)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return

#######################################################################################################################


def _create_generic_directory(parent_dir_path, my_dir_name):

    if parent_dir_path is None:
        parent_dir_path = os.getcwd()

    my_dir_path = os.path.join(parent_dir_path, my_dir_name)

    if os.path.isdir(my_dir_path):
        shutil.rmtree(my_dir_path)

    os.mkdir(my_dir_path)

    return my_dir_path

#######################################################################################################################


def _per_algorithm(models_info, test_set_df):

    aggregations_per_algorithm = {}

    y_true = _get_y_true(test_set_df)

    for model_name in models_info:

        experiments = models_info[model_name]

        aggregations_per_algorithm[model_name] = AggregatedPredictionsClass()

        my_aggregations = aggregations_per_algorithm[model_name]

        for fold, experiment in experiments.items():

            y_pred, scores = experiment.load_predictions(TESTSET)

            my_aggregations.aggregate(y_pred, scores, {fold}, model_name, False)

        my_aggregations.majority_voting()

        my_aggregations.assign_scores(y_true)

        # output_string = "\nAlgorithm = {}".format(model_name) + "\n"
        # output_string += "avrg_all_score = {}".format(str(my_aggregations.avrg_all_scores)) + "\n"
        # output_string += "best_all_score = {}".format(str(my_aggregations.best_all_scores)) + "\n"
        #
        # if my_aggregations.n_best_thr != 0:
        #     output_string += "n_best_thr = {}".format(str(my_aggregations.n_best_thr)) + "\n"
        #     output_string += "best_models = {}".format(str(my_aggregations.best_models)) + "\n"
        #     output_string += "avrg_thr_score = {}".format(str(my_aggregations.avrg_thr_scores)) + "\n"
        #     output_string += "best_thr_score = {}".format(str(my_aggregations.best_thr_scores))
        #
        # print(output_string)

    return aggregations_per_algorithm

#######################################################################################################################


def _algorithm_combinations(models_info, test_set_df, aggregations_per_algorithm):

    y_true = _get_y_true(test_set_df)

    combined_aggregations = {}

    for model_combination in list(_powerset(models_info.keys())):

        if len(model_combination) == 0:
            continue

        model_combination_str = MODEL_DELIMITER.join(model_combination)

        combined_aggregations[model_combination_str] = AggregatedPredictionsClass()

        my_aggregations = combined_aggregations[model_combination_str]

        for model_name in model_combination:

            algorithm_aggregations = aggregations_per_algorithm[model_name]

            if algorithm_aggregations.n_best_thr != 0:
                my_aggregations.aggregate(algorithm_aggregations.avrg_thr_preds,
                                          algorithm_aggregations.avrg_thr_scores,
                                          algorithm_aggregations.best_models[model_name],
                                          model_name,
                                          True)

        if my_aggregations.n_all == 0:
            continue

        my_aggregations.majority_voting()

        my_aggregations.assign_scores(y_true)

        output_string = "\nCombined Algorithms = {}".format(model_combination) + "\n"
        output_string += "Best folds = {}".format(str(my_aggregations.best_models)) + "\n"
        bacc = round(my_aggregations.avrg_all_scores[0], 2)
        accr = round(my_aggregations.avrg_all_scores[1], 2)
        output_string += "Majority voting scores :: BACC = {0}, ACCR = {1}".format(bacc, accr)

        if my_aggregations.avrg_all_preds is not None:
            y_pred = my_aggregations.avrg_all_preds
            scores = my_aggregations.avrg_all_scores
            _plot_results(y_true, y_pred, scores, list(model_combination))
            _plot_confusion_matrix(y_true, y_pred, model_combination)

        print(output_string)

    return combined_aggregations

#######################################################################################################################


def _store_best_aggregations_per_algorithm(aggregations_per_algorithm, models_info, aggpred_dir_path):

    # print("\n~~~~~ Store best aggregations per algorithm ~~~~~")

    bpa_dir_path = _create_generic_directory(aggpred_dir_path, BEST_PER_ALGORITHM)

    for model_name in aggregations_per_algorithm:

        algo_dir_path = _create_generic_directory(bpa_dir_path, model_name)

        my_aggregations = aggregations_per_algorithm[model_name]

        if model_name not in my_aggregations.best_models:
            continue

        best_model_nums = my_aggregations.best_models[model_name]

        for model_num in best_model_nums:

            src_parent_dir_path = models_info[model_name][model_num].best_optuna_trial_path

            src_name = os.listdir(src_parent_dir_path)[0]

            src_path = os.path.join(src_parent_dir_path, src_name)

            shutil.copy(src_path, algo_dir_path)

            trg_path = os.path.join(algo_dir_path, src_name)

            new_name = str(model_num) + MODEL_DELIMITER + src_name

            new_path = os.path.join(algo_dir_path, new_name)

            os.rename(trg_path, new_path)

    return

#######################################################################################################################


def _store_combined_aggregations(combined_aggregations, models_info, aggpred_dir_path):

    # ------------------------------------------------------------------------------------------------------------------

    combagg_dir_path = _create_generic_directory(aggpred_dir_path, COMBINED_ALGORITHMS)

    # ------------------------------------------------------------------------------------------------------------------

    # print("\n~~~~~ Store all combined aggregations ~~~~~")

    allcomb_dir_path = _create_generic_directory(combagg_dir_path, ALL_COMBINATIONS)

    best_score = 0.0

    best_model_combination = None

    best_model_combination_dir_path = None

    for model_combination in combined_aggregations:

        n_models = len(model_combination.split(MODEL_DELIMITER))

        my_aggregations = combined_aggregations[model_combination]

        if n_models != len(my_aggregations.best_models):
            continue

        mc_dir_path = _create_generic_directory(allcomb_dir_path, model_combination)

        for model_name in my_aggregations.best_models:

            best_model_nums = my_aggregations.best_models[model_name]

            for model_num in best_model_nums:

                src_parent_dir_path = models_info[model_name][model_num].best_optuna_trial_path

                src_name = os.listdir(src_parent_dir_path)[0]

                src_path = os.path.join(src_parent_dir_path, src_name)

                # print("\nCopy file\n" + src_path + "\nto\n" + mc_dir_path)

                shutil.copy(src_path, mc_dir_path)

                trg_path = os.path.join(mc_dir_path, src_name)

                new_name = model_name + MODEL_DELIMITER + str(model_num) + MODEL_DELIMITER + src_name

                new_path = os.path.join(mc_dir_path, new_name)

                # print("\nRename file\n" + trg_path + "\nto\n" + new_path)

                os.rename(trg_path, new_path)

        scores = my_aggregations.avrg_all_scores

        if scores[0] > best_score:

            best_score = scores[0]

            best_model_combination = model_combination

            best_model_combination_dir_path = mc_dir_path

    # ------------------------------------------------------------------------------------------------------------------

    if best_model_combination is not None:

        # print("\n~~~~~ Store best combined aggregations ~~~~~")

        isolated_combination_dir_path = os.path.join(combagg_dir_path, BEST_COMBINATION, best_model_combination)
        if os.path.isdir(isolated_combination_dir_path):
            shutil.rmtree(isolated_combination_dir_path)

        shutil.copytree(best_model_combination_dir_path, isolated_combination_dir_path)

        print("\n#####################################################################")
        print("Best model is located under " + isolated_combination_dir_path)
        print("#####################################################################\n")

    # ------------------------------------------------------------------------------------------------------------------

    return

#######################################################################################################################


def execute(models_info, test_set_df):

    # ------------------------------------------------------------------------------------------------------------------

    aggpred_dir_path = _create_generic_directory(None, AGGREGATED_PREDICTIONS)

    # ------------------------------------------------------------------------------------------------------------------

    aggregations_per_algorithm = _per_algorithm(models_info, test_set_df)

    _store_best_aggregations_per_algorithm(aggregations_per_algorithm, models_info, aggpred_dir_path)

    # ------------------------------------------------------------------------------------------------------------------

    combined_aggregations = _algorithm_combinations(models_info, test_set_df, aggregations_per_algorithm)

    _store_combined_aggregations(combined_aggregations, models_info, aggpred_dir_path)

    # ------------------------------------------------------------------------------------------------------------------

    return

#######################################################################################################################
