"""
Optuna example that optimizes a classifier configuration for IMDB movie review dataset.
This script is based on the example of allentune (https://github.com/allenai/allentune).

In this example, we optimize the validation accuracy of
sentiment classification using an AllenNLP jsonnet config file.
Since it is too time-consuming to use the training dataset,
we here use the validation dataset instead.

"""

import argparse
import os
import shutil

import optuna
from optuna.integration.allennlp import dump_best_config
from optuna.integration import AllenNLPExecutor
from packaging import version

import allennlp


# This path trick is used since this example is also
# run from the root of this repository by CI.
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_CONFIG_PATH = "best_classifier.json"


def create_objective(config_file_name, model_dir):
    def objective(trial):
        trial.suggest_float("DROPOUT", 0.0, 1.0)
        trial.suggest_int("EMBEDDING_DIM", 16, 512)
        trial.suggest_int("MAX_FILTER_SIZE", 3, 6)
        trial.suggest_int("NUM_FILTERS", 16, 256)
        trial.suggest_int("HIDDEN_SIZE", 16, 256)

        config_path = os.path.join(EXAMPLE_DIR, config_file_name)
        serialization_dir = os.path.join(model_dir, "test_{}".format(trial.number))
        executor = AllenNLPExecutor(trial, config_path, serialization_dir, force=True)

        return executor.run()
    return objective


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--device", nargs="+", default=[-1], type=int)
    args = parser.parse_args()

    if len(args.device) == 1:
        config_file_name = "classifier.jsonnet"
        model_dir = "result_single"
        study_name = "allennlp_jsonnet_single"
        os.environ["CUDA_DEVICE"] = str(args.device[0])
    else:
        config_file_name = "classifier_distributed.jsonnet"
        model_dir = "result_multi"
        study_name = "allennlp_jsonnet_multi"
        os.environ["CUDA_DEVICES"] = str(args.device)

    if version.parse(allennlp.__version__) < version.parse("2.0.0"):
        raise RuntimeError(
            "`allennlp>=2.0.0` is required for this example."
            " If you want to use `allennlp<2.0.0`, please install `optuna==2.5.0`"
            " and refer to the following example:"
            " https://github.com/optuna/optuna/blob/v2.5.0/examples/allennlp/allennlp_jsonnet.py"
        )

    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///allennlp.db",
        pruner=optuna.pruners.HyperbandPruner(),
        sampler=optuna.samplers.TPESampler(seed=10),
        study_name=study_name,
        load_if_exists=True,
    )

    objective = create_objective(config_file_name, model_dir)
    # study.optimize(objective, n_trials=50, timeout=600)
    study.optimize(objective, n_trials=40)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    dump_best_config(os.path.join(EXAMPLE_DIR, config_file_name), BEST_CONFIG_PATH, study)
    print("\nCreated optimized AllenNLP config to `{}`.".format(BEST_CONFIG_PATH))

    shutil.rmtree(model_dir)
