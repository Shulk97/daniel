# This file is under a custom Research Usage Only (RUO) license.
# Please refer to the license file LICENSE for more details.
import os
import sys

import click
from torch.optim import Adam

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.dirname(PARENT_DIR))
sys.path.append(os.path.dirname(os.path.dirname(PARENT_DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(PARENT_DIR))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PARENT_DIR)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PARENT_DIR))))))

from basic.encoders import FCN_Encoder_Widther
from basic.utils import init_wandb

from conf.conf_utils import merge_config_dicts, complete_dataset_params
from conf.training.base import default_training_cfg
from conf.dataset.base import default_dataset_config_factory
from conf.model.base import default_model_cfg

from OCR.document_OCR.daniel.trainer_daniel import run
from OCR.document_OCR.daniel.iam.compatible_fonts_iam import hw_extended_valid_fonts as valid_hw_extended_iam_fonts, printed_valid_fonts as valid_printed_iam_fonts

@click.command()
@click.option('--mode', default='eval', help='Execution mode, should be train, synth or eval')
def main(mode):
    dataset_name = "IAM"
    dataset_level = "page"
    dataset_variant = "_sem-150dpi-pg_named_entities"

    specific_dataset_cfg = {
        "tagging_mode":'after',
        "config": {
            "layout_tokens_mode": 'MULTI',
            "labels_name": "formatted-18-entities-daniel-charset-custom-split.pkl", # was not used at the beginning because of the bug
            "other_samples":'huggingface:Datasets/raw/wiki_with_NEs_filtered',
            "mean": [235.74465752],
            "std": [44.38831295],
            "height_divisor": 32,  # Image height will be divided by 32
            "synthetic_data": {
                "init_proba": 1.0,# begin proba to generate synthetic document
                "end_proba": 0.2,# end proba to generate synthetic document
                "num_steps_proba": 300e3,# linearly decrease the percent of synthetic document from 90% to 20% through 300000 samples
                "curr_step": 1,# interval to increase the number of lines for curriculum learning
                # config for synthetic line generation
                "config": {
                    'iam': {
                        'one_font_per_pg': True,
                        'one_size_per_pg' : True,
                        'end_at_dot': True,
                        'single_pg': True,
                    },
                    'iam_valid_fonts_list': {
                        'hw': valid_hw_extended_iam_fonts,
                        'printed': valid_printed_iam_fonts,
                        'hw-proba': 0.8
                    },
                },
            },
        },
    }

    specific_model_cfg = {
        "models": {
            "encoder": FCN_Encoder_Widther
        },
        "use_checkpointing": False,
        "transfer_learning": {
            # model_name: [state_dict_name, checkpoint_path, learnable, strict]
            "encoder": ["encoder", "outputs/daniel_iam_ner_strategy_A_custom_split/checkpoints/best-IAM_NER_165.pt", True, True],
            "decoder": ["decoder", "outputs/daniel_iam_ner_strategy_A_custom_split/checkpoints/best-IAM_NER_165.pt", True, False],
        },
    }

    specific_training_cfg = {
        "output_folder": "daniel_iam_ner_strategy_C_custom_split",  # folder name for checkpoint and results
        "max_nb_epochs": 50000,  # maximum number of epochs before to stop
        "load_epoch": "last",# ["best", "last"]: last to continue training, best to evaluate
        "batch_size": 4,  # mini-batch size for training
        "valid_batch_size": 4,  # mini-batch size for validation
        "test_batch_size": 4,  # mini-batch size for validation
        "optimizers": {
            "all": {
                "class": Adam,
                "args": {
                    "lr": 1e-5,
                    "amsgrad": False,
                }
            },
        },
        "eval_on_valid": True,  # Whether to eval and logs metrics on validation set during training or not
        "eval_on_valid_interval": 5,# Interval (in epochs) to evaluate during training
        "focus_metric": "cer",  # Metrics to focus on to determine best epoch
        "expected_metric_value": "low",  # ["high", "low"] What is best for the focus metric value
        "set_name_focus_metric": "{}-valid".format(dataset_name),  # Which dataset to focus on to select best weights
        "train_metrics": ["loss_ce", "cer", "wer", "syn_max_lines","cer-ner"],  # Metrics name for training
        "eval_metrics": ["cer", "wer",'f1','cer-ner'],#, "map_cer"],  # Metrics name for evaluation on validation set during training
        "force_cpu": False,  # True for debug purposes
        "max_char_prediction": 900,
        "max_char_inference": 300,  # max number of token prediction
        "teacher_forcing_scheduler": {
            "min_error_rate": 0.3,
            "max_error_rate": 0.3,
            "total_num_steps": 5e6
        },
        "use_wandb": False,
    }

    params = {}

    params['model_params'] = merge_config_dicts(default_model_cfg, specific_model_cfg)

    default_dataset_cfg = default_dataset_config_factory(dataset_name, dataset_level, dataset_variant)
    params['dataset_params'] = merge_config_dicts(default_dataset_cfg, specific_dataset_cfg)
    params['training_params'] = merge_config_dicts(default_training_cfg, specific_training_cfg)

    params['dataset_params'] = complete_dataset_params(params['dataset_params'], params['model_params'], params['training_params'])

    params["model_params"]["max_char_prediction"] = params["training_params"]["max_char_prediction"]
    if params["training_params"]["use_wandb"]:
        init_wandb(projet_name="daniel", exp_id='iam-ner-custom-split-strat-C', params=params, dataset_name=dataset_name)

    run(params, mode=mode, dataset_names=[dataset_name], metrics=["cer", "wer", 'f1'], set_names=["test"])

if __name__ == "__main__":
    main()