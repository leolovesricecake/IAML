import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from runs.run_cli import apply_cli_args, build_experiment_name_prefix, parse_args


def main(argv=None):
    args = parse_args(argv)
    apply_cli_args(args)

    from main.hp_search import HpSearch
    from main.run_fine_tune import FineTune
    from main.run_infrence_pre_train import InferencePretrain
    from main.run_pre_train import PreTrain
    from models.train_models_utils import load_explained_model
    from config.config import ExpArgs
    from utils.utils_functions import get_current_time

    print(
        "*" * 20,
        args.task,
        args.explained_model_backbone,
        args.interpreter_model_backbone,
        args.metric,
        "*" * 20,
        flush=True,
    )

    time_str = get_current_time()
    experiment_name_prefix = build_experiment_name_prefix()

    explained_model = load_explained_model()

    hp_experiment_name = f"HP_{experiment_name_prefix}_{time_str}"
    hp = HpSearch(hp_experiment_name, explained_model=explained_model).run()

    pre_train_experiment_name = f"PRETRAIN_{experiment_name_prefix}_{time_str}"
    pretrain_model_path = PreTrain(hp, pre_train_experiment_name, explained_model=explained_model).run()

    ExpArgs.fine_tuned_interpreter_model_path = pretrain_model_path

    inference_pretrain_experiment_name = f"INFERENCE_PRETRAIN_{experiment_name_prefix}_{time_str}"
    InferencePretrain(hp, inference_pretrain_experiment_name, explained_model=explained_model).run()

    fine_tune_exp_name = f"FINE_TUNE_{experiment_name_prefix}_{time_str}"
    FineTune(hp, fine_tune_exp_name, explained_model=explained_model).run()
    print(
        "*" * 20,
        "END OF ",
        args.task,
        args.explained_model_backbone,
        args.interpreter_model_backbone,
        args.metric,
        "*" * 20,
    )


if __name__ == "__main__":
    main()
