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
from conf.dataset.base import default_dataset_config_factory, max_nb_lines
from conf.model.base import default_model_cfg

from OCR.document_OCR.daniel.trainer_daniel import run
from OCR.document_OCR.daniel.READ.compatible_fonts_read import valid_fonts as read_valid_fonts, hw_fonts_wiki
from OCR.document_OCR.daniel.RIMES.compatible_fonts_rimes import valid_fonts as rimes_valid_fonts
from OCR.document_OCR.daniel.iam.compatible_fonts_iam import hw_extended_valid_fonts as valid_hw_extended_iam_fonts, printed_valid_fonts as valid_printed_iam_fonts
from OCR.document_OCR.daniel.RIMES.compatible_fonts_rimes import valid_hw_extended_fonts as valid_hw_extended_rimes_fonts, valid_printed_fonts as valid_printed_rimes_fonts



@click.command()
@click.option('--mode', default='eval', help='Execution mode, should be train, synth or eval')
def main(mode):
    dataset_names = ['IAM', 'RIMES', 'READ_2016', 'IAM_NER']
    dataset_name = "MULTI_LINGUAL"
    dataset_level = "page"
    dataset_variant = "_sem_named_entities"

    specific_dataset_cfg = {
        "tagging_mode":'after-no-hierarchy_exopopp',
            "tagging_modes":{
                'EXOPOPP_NER': 'after-no-hierarchy_exopopp',
                'IAM_NER': 'after',
            },
        "datasets": {
            dataset_name: "Datasets/formatted/MULTI_LINGUAL_page_sem_named_entities",
        },
        "valid": {
            "IAM-valid": [('MULTI_LINGUAL', "valid-IAM"), ],
            "RIMES-valid": [('MULTI_LINGUAL', "valid-RIMES"), ],
            "READ_2016-valid": [('MULTI_LINGUAL', "valid-READ_2016"), ],
            "IAM_NER-valid": [('MULTI_LINGUAL', "valid-IAM_NER"), ],
        },
        'synth_eval_loaders': ['IAM', 'RIMES', 'READ_2016', 'IAM_NER'],
        'add_NEs_in_charset':True,
        "config": {
            "tagging_mode":'after',
            "labels_name": "labels-multi-ner.pkl",
            "mean" :[235.74465752],
            "std" : [44.38831295],
            'multi_sampling': True,
            'multi_sampling_weights': {
                'IAM': 1/(747*4), # 8.25%
                'RIMES': 1/1050,
                'READ_2016': 1/350,
                'IAM_NER': 3/(747*4), # 24.75%
                'EXOPOPP': 1/(250*4),
                'EXOPOPP_NER':  3/(250*4),
            },
            'index_datasets': {
                'IAM': [0, 746],
                'RIMES': [747, 1796],
                'READ_2016': [1797, 2146],
                'IAM_NER': [2147, 2893],
            },
            'read_sampling_weights': {
                'wiki_de':100,
            },
            "multi_samples":{
                'IAM':'Datasets/raw/wiki_en',
                'IAM_NER':'Datasets/raw/wiki_en_ner',
                'RIMES':'Datasets/raw/wiki_fr',
                'READ_2016': ['Datasets/raw/wiki_de.txt']
            },
            "constraints": ['use_subwords_bart','use_subwords', "mono_start"],
            "height_divisor": 32,  # Image height will be divided by 32
            "synthetic_data": {
                'MULTI_synth_props': { # with transfer learnig from multilingual synth
                    'IAM': 10, # 10%
                    'RIMES': 20, # 45%
                    'READ_2016': 18, # 20%
                    'IAM_NER': 52, # 25%
                },
                "max_nb_lines_READ_2016": max_nb_lines['READ_2016'],  # maximum number of lines for curriculum learning
                "max_nb_lines_IAM": max_nb_lines['IAM'],  # maximum number of lines for curriculum learning
                "max_nb_lines_IAM_NER": max_nb_lines['IAM_NER'],  # maximum number of lines for curriculum learning
                "max_nb_lines_RIMES": max_nb_lines['RIMES'],  # maximum number of lines for curriculum learning
                'datasets' : ['IAM', 'RIMES', 'READ_2016', 'IAM_NER'],
                "init_proba": 1.0,# begin proba to generate synthetic document
                "end_proba": 0.2,# end proba to generate synthetic document
                "num_steps_proba": 300e3,# linearly decrease the percent of synthetic document from 90% to 20% through 300000 samples
                "curr_step": 1,# interval to increase the number of lines for curriculum learning
                # config for synthetic line generation
                "config": {
                    'max_pad_left_ratio':0.75,
                    'read': {
                        'one_font_per_pg': True,
                        'one_size_per_pg' : True,
                        'end_at_dot': True
                    },
                    'iam': {
                        'one_font_per_pg': True,
                        'one_size_per_pg' : True,
                        'end_at_dot': True,
                        'single_pg': True,
                    },
                    'rimes': {
                        'one_font_per_pg': True,
                        'one_size_per_pg' : True,
                        'end_at_dot': True
                    },
                    'rimes_valid_fonts_list': {
                        'hw': valid_hw_extended_rimes_fonts,
                        'printed': valid_printed_rimes_fonts,
                        'hw-proba': 0.8
                    },
                    'iam_valid_fonts_list': {
                        'hw': valid_hw_extended_iam_fonts,
                        'printed': valid_printed_iam_fonts,
                        'hw-proba': 0.8
                    },
                    'read_valid_fonts_list': {
                        'wiki': hw_fonts_wiki,
                    },
                    'read_valid_fonts': read_valid_fonts,
                    'rimes_valid_fonts': rimes_valid_fonts,
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
            "encoder": ["encoder", "outputs/daniel_multi_synth/checkpoints/last_1875.pt", True, True],
            "decoder": ["decoder", "outputs/daniel_multi_synth/checkpoints/last_1875.pt", True, False],
        },
    }

    specific_training_cfg = {
        "output_folder": "daniel_iam_ner_strategy_A_rwth_split",  # folder name for checkpoint and results
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
        "focus_metrics": {
            'IAM': 'cer',
            'RIMES': 'cer',
            'READ_2016': 'cer',
            'IAM_NER': 'f1',
        },
        "expected_metric_values": {
            'IAM': 'low',
            'RIMES': 'low',
            'READ_2016': 'low',
            'IAM_NER': 'high',
        },
        "expected_metric_value": "low",  # ["high", "low"] What is best for the focus metric value
        "set_name_focus_metrics": ["{}-valid".format(ds_name) for ds_name in dataset_names],  # Which dataset to focus on to select best weights
        "train_metrics": ["loss_ce", "cer", "wer", "syn_max_lines", 'cer-ner'],  # Metrics name for training
        "eval_metrics": ["cer", "wer", 'f1', 'stricter_f1'],
        "force_cpu": False,  # True for debug purposes
        "max_char_prediction": 900,
        "max_char_inferences": {
            'IAM': 250,
            'RIMES': 900,
            'READ_2016': 400,
            'IAM_NER': 300,
        },
        "teacher_forcing_scheduler": {
            "min_error_rate": 0.3,
            "max_error_rate": 0.3,
            "total_num_steps": 5e6
        },
        "use_wandb": False,
    }

    params = {}

    params['model_params'] = merge_config_dicts(default_model_cfg, specific_model_cfg)

    default_dataset_cfg = default_dataset_config_factory(dataset_name, dataset_level, dataset_variant, multi=True)
    params['dataset_params'] = merge_config_dicts(default_dataset_cfg, specific_dataset_cfg)
    params['training_params'] = merge_config_dicts(default_training_cfg, specific_training_cfg)

    params['dataset_params'] = complete_dataset_params(params['dataset_params'], params['model_params'], params['training_params'])

    params["model_params"]["max_char_prediction"] = params["training_params"]["max_char_prediction"]
    if params["training_params"]["use_wandb"]:
        init_wandb(projet_name="daniel", exp_id='iam-ner-rwth-split-strat-A', params=params, dataset_name=dataset_name)

    run(params, mode=mode, dataset_names=dataset_names, set_names=["test"], multi=True)

if __name__ == "__main__":
    main()