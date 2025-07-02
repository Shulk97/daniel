# This file is under a custom Research Usage Only (RUO) license.
# Please refer to the license file LICENSE for more details.
import os
import sys
import json
from pathlib import Path

import tqdm
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.dirname(PARENT_DIR))
sys.path.append(os.path.dirname(os.path.dirname(PARENT_DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(PARENT_DIR))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PARENT_DIR)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PARENT_DIR))))))

from basic.encoders import FCN_Encoder_WidtherFeature
from basic.generic_dataset_manager import load_unlabelled_img_batch
from basic.utils import create_fake_batch_names

from conf.conf_utils import merge_config_dicts, complete_dataset_params
from conf.training.base import default_training_cfg
from conf.dataset.base import default_dataset_config_factory
from conf.model.base import default_model_cfg

from OCR.document_OCR.daniel.trainer_daniel import Manager

def inference_on_unlabelled_dataset(folder_path, inference_params, path_pred, max_length=2000, start_token=0):
    global_pred_dict = {}

    batch_file_lists = create_fake_batch_names(folder_path, 1)

    for batch_file_list in tqdm.tqdm(batch_file_lists):
        batch_data = load_unlabelled_img_batch(batch_file_list, inference_params['dataset_params'])

        batch_pred = model.predict_one_unlabelled_image(batch_data, max_length=max_length, start_token=start_token)
        for img_path, img_pred_dict in zip(batch_file_list, batch_pred['str_x']):
            global_pred_dict[img_path] = {'pred': img_pred_dict}

    with open(Path(path_pred).with_suffix('.json'), "w") as f:
        json.dump(global_pred_dict, f)


def inference_on_unlabelled_image(image_path, inference_params, max_length=2000, start_token=0):
    batch_data = load_unlabelled_img_batch([image_path], inference_params['dataset_params'])
    batch_pred = model.predict_one_unlabelled_image(batch_data, max_length=max_length, start_token=start_token)

    return batch_pred['str_x']

dataset_name = "MULTI_LINGUAL"
dataset_level = "page"
dataset_variant = "_sem_named_entities"

specific_dataset_cfg = {
    "tagging_mode":'after-no-hierarchy_exopopp',
    "tagging_modes":{
        'EXOPOPP_NER': 'after-no-hierarchy_exopopp',
        'IAM_NER': 'after',

    },
    "config": {
        "layout_tokens_mode": 'MULTI_EXOPOPP',
        "mean" : [219.16383805],
        "std" : [56.89969272],
        "labels_name": "labels-multi-ner-mpopp.pkl",
        "height_divisor": 16,  # Image height will be divided by 16
        "constraints": ['use_subwords_bart','use_subwords', "mono_start"],
        'EXOPOPP_charset': ['\n', ' ', '"', '#', '&', "'", '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '°', 'Â', 'à', 'â', 'ä', 'ç', 'è', 'é', 'ê', 'ë', 'î', 'ï', 'ô', 'ö', 'ù', 'û', 'ü', '⌚', '⌛', '⏰', '⏳', 'Ⓑ', 'Ⓘ', 'Ⓜ', 'Ⓝ', 'Ⓟ', 'Ⓢ', 'ⓑ', 'ⓘ', 'ⓜ', 'ⓝ', 'ⓟ', 'ⓢ', '⚓', '⚰', '🁴', '🂂', '🃄', '🃌', '🃑', '🄑', '🄰', '🄶', '🄻', '🄾', '🅆', '🅍', '🆀', '🆉', '🇦', '🇧', '🇨', '🇩', '🇪', '🇬', '🇭', '🇮', '🇯', '🇰', '🇱', '🇲', '🇳', '🇴', '🇵', '🇶', '🇷', '🇺', '🇻', '🇼', '🇽', '🇾', '🇿', '🌂', '🌇', '🌈', '🌉', '🌌', '🌍', '🌐', '🌝', '🌞', '🌠', '🌥', '🌲', '🌴', '🍁', '🍂', '🍆', '🍎', '🍒', '🍗', '🍜', '🍞', '🍭', '🍲', '🍽', '🎁', '🎉', '🎊', '🎐', '🎜', '🎠', '🎤', '🎩', '🎸', '🎺', '🎼', '🏁', '🏉', '🏓', '🏠', '🏡', '🏥', '🏰', '🏳', '🏶', '🐁', '🐃', '🐅', '🐋', '🐑', '🐓', '🐕', '🐙', '🐡', '🐢', '🐮', '🐱', '👋', '👒', '👔', '👕', '👖', '👘', '👠', '👦', '👧', '👨', '👰', '👴', '👵', '👶', '👹', '👼', '💈', '💚', '💛', '💞', '💢', '💣', '💦', '💬', '💭', '💯', '💲', '📅', '📆', '📌', '📎', '📕', '📖', '📗', '📺', '🔂', '🔍', '🔎', '🔑', '🔒', '🔗', '🔟', '🔠', '🔡', '🔢', '🔥', '🔧', '🔱', '🕈', '🕍', '🕎', '🕑', '🕘', '🕮', '🖅', '🖔', '🖘', '🖢', '🖦', '🖱', '🖶', '🖺', '🖻', '🗓', '🗘', '🗨', '🗯', '🗴', '🗺', '😀', '😁', '😍', '😎', '😔', '😚', '😠', '😡', '😢', '😭', '😶', '🙀', '🙃', '🙈', '🚀', '🚌', '🚑', '🚔', '🚲', '🚳', '🚾', '🛌', '🛣', '🛤', '🛪', '🠔', '🠖', '🠱', '🠳', '🥑', '🥦', '🥸', '🧐', '🧣', '🧤', '🪛', '🪦'],
    },
}

specific_model_cfg = {
    "models": {
        "encoder": FCN_Encoder_WidtherFeature,
    },
    "use_checkpointing": True,
    "transfer_learning": None,
}

specific_training_cfg = {
    "output_folder": "daniel_mpopp_strategy_A",  # folder name for checkpoint and results
    "load_epoch": "best",
    "batch_size": 1,  # mini-batch size for training
    "valid_batch_size": 1,  # mini-batch size for validation
    "test_batch_size": 1,  # mini-batch size for validation
    "force_cpu": False,  # True for debug purposes
    "max_char_prediction": 2300,#4300,  # max number of token prediction
    "max_char_inference": 2000,#4300,  # max number of token prediction
    "max_char_inferences": {
        'IAM': 250,
        'RIMES': 900,
        'READ_2016': 400,
        'IAM_NER': 300,
        'EXOPOPP': 2200,
        'EXOPOPP_NER': 2200,
    },
}

params = {"inference_mode": True}

params['model_params'] = merge_config_dicts(default_model_cfg, specific_model_cfg)

default_dataset_cfg = default_dataset_config_factory(dataset_name, dataset_level, dataset_variant)
params['dataset_params'] = merge_config_dicts(default_dataset_cfg, specific_dataset_cfg)
params['training_params'] = merge_config_dicts(default_training_cfg, specific_training_cfg)

params['dataset_params'] = complete_dataset_params(params['dataset_params'], params['model_params'], params['training_params'])

params["model_params"]["max_char_prediction"] = params["training_params"]["max_char_prediction"]

params["dataset_params"]["reduce_dims_factor"] = np.array([
    params["dataset_params"]["config"]["height_divisor"],
    params["dataset_params"]["config"]["width_divisor"],
    1
])

params["model_params"]['vocab'] = params["dataset_params"]['subword_tokenizer'].get_vocab()
params["model_params"]["max_char_prediction"] = params["training_params"]["max_char_prediction"]

params['model_params']['subword_tokenizer'] = params["dataset_params"]['subword_tokenizer']
params['model_params']['subword_tokenizer'].__name__ = 'subword_tokenizer'

model = Manager(params)
model.load_model()
