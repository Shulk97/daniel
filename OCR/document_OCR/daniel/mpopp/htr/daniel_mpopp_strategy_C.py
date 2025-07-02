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

from basic.encoders import FCN_Encoder_WidtherFeature
from basic.utils import init_wandb

from conf.conf_utils import merge_config_dicts, complete_dataset_params
from conf.training.base import default_training_cfg
from conf.dataset.base import default_dataset_config_factory
from conf.model.base import default_model_cfg

from OCR.document_OCR.daniel.trainer_daniel import run

@click.command()
@click.option('--mode', default='eval', help='Execution mode, should be train, synth or eval')
def main(mode):
    dataset_name = "EXOPOPP_global"
    dataset_level = "page"
    dataset_variant = "_sem"

    specific_dataset_cfg = {
        "config": {
            "layout_tokens_mode": 'MULTI',
            "mean" : [182.72022651],
            "std" : [63.95714202],
            "labels_name": "labels-mpopp.pkl",
            "other_samples":'huggingface:Datasets/raw/wiki-fr-multi-bis',
            'EXOPOPP_charset': ['\n', ' ', '"', '#', '&', "'", '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '°', 'Â', 'à', 'â', 'ä', 'ç', 'è', 'é', 'ê', 'ë', 'î', 'ï', 'ô', 'ö', 'ù', 'û', 'ü', '⌚', '⌛', '⏰', '⏳', 'Ⓑ', 'Ⓘ', 'Ⓜ', 'Ⓝ', 'Ⓟ', 'Ⓢ', 'ⓑ', 'ⓘ', 'ⓜ', 'ⓝ', 'ⓟ', 'ⓢ', '⚓', '⚰', '🁴', '🂂', '🃄', '🃌', '🃑', '🄑', '🄰', '🄶', '🄻', '🄾', '🅆', '🅍', '🆀', '🆉', '🇦', '🇧', '🇨', '🇩', '🇪', '🇬', '🇭', '🇮', '🇯', '🇰', '🇱', '🇲', '🇳', '🇴', '🇵', '🇶', '🇷', '🇺', '🇻', '🇼', '🇽', '🇾', '🇿', '🌂', '🌇', '🌈', '🌉', '🌌', '🌍', '🌐', '🌝', '🌞', '🌠', '🌥', '🌲', '🌴', '🍁', '🍂', '🍆', '🍎', '🍒', '🍗', '🍜', '🍞', '🍭', '🍲', '🍽', '🎁', '🎉', '🎊', '🎐', '🎜', '🎠', '🎤', '🎩', '🎸', '🎺', '🎼', '🏁', '🏉', '🏓', '🏠', '🏡', '🏥', '🏰', '🏳', '🏶', '🐁', '🐃', '🐅', '🐋', '🐑', '🐓', '🐕', '🐙', '🐡', '🐢', '🐮', '🐱', '👋', '👒', '👔', '👕', '👖', '👘', '👠', '👦', '👧', '👨', '👰', '👴', '👵', '👶', '👹', '👼', '💈', '💚', '💛', '💞', '💢', '💣', '💦', '💬', '💭', '💯', '💲', '📅', '📆', '📌', '📎', '📕', '📖', '📗', '📺', '🔂', '🔍', '🔎', '🔑', '🔒', '🔗', '🔟', '🔠', '🔡', '🔢', '🔥', '🔧', '🔱', '🕈', '🕍', '🕎', '🕑', '🕘', '🕮', '🖅', '🖔', '🖘', '🖢', '🖦', '🖱', '🖶', '🖺', '🖻', '🗓', '🗘', '🗨', '🗯', '🗴', '🗺', '😀', '😁', '😍', '😎', '😔', '😚', '😠', '😡', '😢', '😭', '😶', '🙀', '🙃', '🙈', '🚀', '🚌', '🚑', '🚔', '🚲', '🚳', '🚾', '🛌', '🛣', '🛤', '🛪', '🠔', '🠖', '🠱', '🠳', '🥑', '🥦', '🥸', '🧐', '🧣', '🧤', '🪛', '🪦'],
            "height_divisor": 16,  # Image height will be divided by 16
            "synthetic_data": {
                'use_marge_noms' : True,
                'use_marge_infos' : False,
                "max_nb_lines_marge_noms": 8,
                "max_nb_lines_marge_info": 0,#15,
                "max_nb_lines_paragraphe": 62,
                'force_synth_eval':False,
                'margin_first':False,
                "init_proba": 1.0,# begin proba to generate synthetic document
                "end_proba": 0.2,# end proba to generate synthetic document
                "num_steps_proba": 300e3,# linearly decrease the percent of synthetic document from 90% to 20% through 300000 samples
                "curr_step": 1,# interval to increase the number of lines for curriculum learning
                # config for synthetic line generation
                "config": {
                    'exopopp': {
                        'one_font_per_pg': True,
                        'one_size_per_pg' : True,
                        'end_at_dot': True,
                        'marge_noms_max_len': 25,
                    },
                    'max_pad_left_ratio':0.75,
                    "font_size_mins": {
                        'marge_noms': 30,
                        'marge_info': 24,
                        # 'marge_info': 27,
                        'paragraphe':30
                    },
                    "font_size_maxs": {
                        'marge_noms': 45,
                        'marge_info': 26,
                        # 'marge_info': 30,
                        'paragraphe':45
                    },
                    "padding_bottom_ratio_max": 0.15,
                },
            },
        },
    }

    specific_model_cfg = {
        "models": {
            "encoder": FCN_Encoder_WidtherFeature
        },
        "use_checkpointing": True,
        "transfer_learning": {
            # model_name: [state_dict_name, checkpoint_path, learnable, strict]
            "encoder": ["encoder", "outputs/daniel_mpopp_strategy_A/checkpoints/best-EXOPOPP_167.pt", True, True],
            "decoder": ["decoder", "outputs/daniel_mpopp_strategy_A/checkpoints/best-EXOPOPP_167.pt,", True, False],
        },
    }

    specific_training_cfg = {
        "output_folder": "daniel_mpopp_strategy_C",  # folder name for checkpoint and results
        "max_nb_epochs": 50000,  # maximum number of epochs before to stop
        "load_epoch": "last",# ["best", "last"]: last to continue training, best to evaluate
        "batch_size": 1,  # mini-batch size for training
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
        "train_metrics": ["loss_ce", "cer", "wer", "syn_max_lines"],  # Metrics name for training
        "eval_metrics": ["cer", "wer"],#, "map_cer"],  # Metrics name for evaluation on validation set during training
        "force_cpu": False,  # True for debug purposes
        "max_char_prediction": 2300,
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
        init_wandb(projet_name="daniel", exp_id='mpopp-strat-C', params=params, dataset_name=dataset_name)

    run(params, mode=mode, dataset_names=[dataset_name], metrics=["cer", "wer"], set_names=["test"])

if __name__ == "__main__":
    main()