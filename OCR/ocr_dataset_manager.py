#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#  - Thomas Constum
#
#  This software is a computer program written in Python  whose purpose is to
#  provide public implementation of deep learning works, in pytorch.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.
"""
Module: ocr_dataset_manager.py

This module provides classes for handling OCR/HTR datasets and managing OCR tasks.

Classes:
- OCRDatasetManager: A specific class to handle OCR/HTR tasks.
- OCRDataset: A specific class to handle OCR/HTR datasets.

"""

import bisect
import copy
import json
import os
import pickle
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from skimage import transform as transform_skimage
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import BartTokenizer, XLMRobertaTokenizer

from basic.generic_dataset_manager import (DatasetManager, GenericDataset,
                                           apply_preprocessing)
from basic.transforms import RandomRotation, Tightening, apply_transform
from basic.utils import (pad_image, pad_image_width_random,
                         pad_image_width_right, pad_images, pad_sequences_1D,
                         rand, rand_uniform, randint, resize_max)
from Datasets.dataset_formatters.exo_popp_formatter import (
    MATCHING_NAMED_ENTITY_TOKENS, MATCHING_NAMED_ENTITY_TOKENS_NO_HIERARCHY)
from Datasets.dataset_formatters.exo_popp_formatter import \
    SEM_MATCHING_TOKENS as EXOPOPP_MATCHING_TOKENS
from Datasets.dataset_formatters.iam_formatter import (
    EXO_POPP_MULTI_MATCHING_TOKENS, MULTI_MATCHING_TOKENS,
    MULTI_MATCHING_TOKENS_STR)
from Datasets.dataset_formatters.read2016_formatter import \
    READ_UNSUPPORTED_CHARS
from Datasets.dataset_formatters.read2016_formatter import \
    SEM_MATCHING_TOKENS as READ_MATCHING_TOKENS
from Datasets.dataset_formatters.rimes_formatter import RIMES_UNSUPPORTED_CHARS
from Datasets.dataset_formatters.rimes_formatter import \
    SEM_MATCHING_TOKENS as RIMES_MATCHING_TOKENS
from Datasets.dataset_formatters.rimes_formatter import \
    SEM_MATCHING_TOKENS_STR as RIMES_MATCHING_TOKENS_STR
from Datasets.dataset_formatters.rimes_formatter import \
    order_text_regions as order_text_regions_rimes
from OCR.document_OCR.daniel.synth_doc.synth_doc_generation import (
    add_newlines, generate_english_wiki_line_label, generate_page_text,
    generate_synthetic_exopopp_page, generate_synthetic_iam_mpopp_page)
from OCR.ocr_utils import LM_str_to_ind, LM_str_to_ind_subwords
from OCR.document_OCR.daniel.synth_doc.synth_doc_generation import is_label_invalid

class OCRDatasetManager(DatasetManager):
    """
    Specific class to handle OCR/HTR tasks
    """

    def __init__(self, params):
        super(OCRDatasetManager, self).__init__(params)

        self.charset = params["charset"] if "charset" in params else self.get_merged_charsets()


        if (
            "synthetic_data" in self.params["config"]
            and self.params["config"]["synthetic_data"]
            and "config" in self.params["config"]["synthetic_data"]
        ):
            self.char_only_set = self.charset.copy()
            for token_dict in [ # remove the layout tokens from the char only set
                RIMES_MATCHING_TOKENS,
                READ_MATCHING_TOKENS,
                {**EXOPOPP_MATCHING_TOKENS,**MATCHING_NAMED_ENTITY_TOKENS},
                EXO_POPP_MULTI_MATCHING_TOKENS
            ]:
                for key in token_dict:
                    if key in self.char_only_set:
                        self.char_only_set.remove(key)
                    if token_dict[key] in self.char_only_set:
                        self.char_only_set.remove(token_dict[key])
            for token in [
                "\n",
            ]:
                if token in self.char_only_set:
                    self.char_only_set.remove(token)

            if not 'valid_fonts' in self.params["config"]["synthetic_data"]["config"]:
                fonts_path = self.params["config"]["synthetic_data"]["config"]['fonts_path'] if 'fonts_path' in self.params["config"]["synthetic_data"]["config"] else 'Fonts'
                if "characters_vocab" in params:
                    self.characters_vocab = params["characters_vocab"]
                    self.params["config"]["synthetic_data"]["config"]["valid_fonts"] = get_valid_fonts(
                        self.characters_vocab, fonts_path
                    )
                else:
                    self.params["config"]["synthetic_data"]["config"]["valid_fonts"] = get_valid_fonts(
                        self.char_only_set, fonts_path
                    )

        if "new_tokens" in params:
            self.charset = sorted(list(set(self.charset).union(set(params["new_tokens"]))))

        self.tokens = {
            "pad": params["config"]["padding_token"],
        }
        if 'start_token' in params["config"]:
            self.tokens['start'] = params["config"]["start_token"]
        if 'end_token' in params["config"]:
            self.tokens['end'] = params["config"]["end_token"]

        if self.params["config"]["charset_mode"].lower() == "ctc":
            self.tokens["blank"] = len(self.charset)
            self.tokens["pad"] = self.tokens["pad"] if self.tokens["pad"] else len(self.charset) + 1
            self.params["config"]["padding_token"] = self.tokens["pad"]
        elif self.params["config"]["charset_mode"] == "seq2seq":
            self.tokens["end"] = self.tokens["end"] if "end" in self.tokens else len(self.charset)
            self.tokens["start"] = self.tokens["start"] if 'start' in self.tokens else len(self.charset) + 1
            self.tokens["pad"] = self.tokens["pad"] if ("pad" in self.tokens and self.tokens["pad"]) else len(self.charset) + 2
            self.params["config"]["padding_token"] = self.tokens["pad"]

    def get_merged_charsets(self):
        """
        Merge the charset of the different datasets used
        """
        datasets = self.params["datasets"]
        charset = set()
        for key in datasets.keys():
            with open(os.path.join(datasets[key], self.params['config'].get('labels_name','labels.pkl')), "rb") as f:
                info = pickle.load(f)
                charset = charset.union(set(info["charset"]))
        if "\n" in charset and "remove_linebreaks" in self.params["config"]["constraints"]:
            charset.remove("\n")
        if "" in charset:
            charset.remove("")
        if "use_subwords" in self.params["config"]["constraints"]:
            return list(charset)
        else:
            return sorted(list(charset))

    def apply_specific_treatment_after_dataset_loading(self, dataset):
        dataset.charset = self.charset
        dataset.joined_charset = ''.join(self.charset)

        dataset.tokens = self.tokens
        if dataset.params["config"].get('preprocessed_samples',None) and dataset.set_name == 'train':
            with open(dataset.params["config"]['preprocessed_samples'],'rb') as f:
                dataset.samples = pickle.load(f)
        else:
            dataset.convert_labels()
        if (
            "READ_2016" in dataset.name
            and "augmentation" in dataset.params["config"]
            and dataset.params["config"]["augmentation"]
        ):
            dataset.params["config"]["augmentation"]["fill_value"] = tuple(
                [int(i) for i in dataset.mean]
            )
        if (
            "padding" in dataset.params["config"]
            and dataset.params["config"]["padding"]["min_height"] == "max"
        ):
            dataset.params["config"]["padding"]["min_height"] = max(
                [s["img"].shape[0] for s in self.train_dataset.samples]
            )
        if (
            "padding" in dataset.params["config"]
            and dataset.params["config"]["padding"]["min_width"] == "max"
        ):
            dataset.params["config"]["padding"]["min_width"] = max(
                [s["img"].shape[1] for s in self.train_dataset.samples]
            )



class CustomXLMRobertaTokenizer(XLMRobertaTokenizer):
    def encode(self, text, *args, **kwargs):
        # Replace \n with <\n> before encoding
        text = text.replace('<\n>', '\n').replace('\n', '<\n>')
        return super().encode(text, *args, **kwargs)

    def decode(self, token_ids, *args, **kwargs):
        # Decode normally, then replace <\n> with \n
        decoded_text = super().decode(token_ids, *args, **kwargs)
        return decoded_text.replace('<\n>', '\n')

class OCRDataset(GenericDataset):
    """
    Specific class to handle OCR/HTR datasets
    """

    def __init__(self, params, set_name, custom_name, paths_and_sets):
        super(OCRDataset, self).__init__(params, set_name, custom_name, paths_and_sets)
        self.charset = None
        self.tokens = None
        self.reduce_dims_factor = np.array(
            [params["config"]["height_divisor"], params["config"]["width_divisor"], 1]
        )
        self.collate_function = OCRCollateFunction
        self.synthetic_id = 0

        if 'use_subwords_bart' in self.params['config']['constraints']:
            bart_path = params['bart_path'] if 'bart_path' in params else "naver-clova-ix/donut-base"
            if "subword_tokenizer" in params:
                self.subword_tokenizer = params["subword_tokenizer"]
            elif 'facebook' in bart_path:
                self.subword_tokenizer = BartTokenizer.from_pretrained(bart_path)
            else:
                #
                if params.get('use_special_line_break', False):
                    self.subword_tokenizer = CustomXLMRobertaTokenizer.from_pretrained(bart_path)
                    self.subword_tokenizer.add_special_tokens({'additional_special_tokens': ['<\n>']})
                else:
                    self.subword_tokenizer = XLMRobertaTokenizer.from_pretrained(bart_path)
                    if 'use_line_break'in params and params['use_line_break']:
                        self.subword_tokenizer.add_tokens(["\n"])
                if 'add_layout_tokens_in_charset' in params and params['add_layout_tokens_in_charset']:
                    # add layouts tokens to the vocabulary of the tokenizer
                    if 'MULTI_LINGUAL' in params['datasets'] or ('layout_tokens_mode' in params['config'] and 'MULTI' in params['config']['layout_tokens_mode']):
                        tokens_to_add = list(MULTI_MATCHING_TOKENS.values())+list(MULTI_MATCHING_TOKENS.keys()) + ['<s-IAM>','<s-RIMES>','<s-READ>'] + READ_UNSUPPORTED_CHARS
                        if 'tagging_mode' in params:
                            tokens_to_add += ['<s-IAM_NER>']
                            if 'EXOPOPP_global' in params['datasets'] or ('layout_tokens_mode' in params['config'] and 'EXOPOPP' in params['config']['layout_tokens_mode']):
                                tokens_to_add += ['<s-EXOPOPP_NER>']
                        if ('synth_eval_loaders' in params and "EXOPOPP" in params['synth_eval_loaders']) or 'EXOPOPP_global' in params['datasets']:
                            tokens_to_add += list(EXO_POPP_MULTI_MATCHING_TOKENS.values())+list(EXO_POPP_MULTI_MATCHING_TOKENS.keys())
                        if params['config'].get('rimes_extra_subwords',False):
                            tokens_to_add += RIMES_UNSUPPORTED_CHARS
                    elif 'READ_2016' in params['datasets']:
                        tokens_to_add = list(READ_MATCHING_TOKENS.values())+list(READ_MATCHING_TOKENS.keys()) + READ_UNSUPPORTED_CHARS
                    elif 'RIMES' in params['datasets']:
                        tokens_to_add = list(RIMES_MATCHING_TOKENS.values())+list(RIMES_MATCHING_TOKENS.keys())
                        if params['config'].get('rimes_extra_subwords',False):
                            tokens_to_add += RIMES_UNSUPPORTED_CHARS
                    elif 'EXOPOPP_global' in params['datasets']:
                        tokens_to_add = list(EXO_POPP_MULTI_MATCHING_TOKENS.values())+list(EXO_POPP_MULTI_MATCHING_TOKENS.keys())
                        if 'named_entities' in params['datasets']['EXOPOPP_global']:
                            tokens_to_add += ['<s-EXOPOPP_NER>']
                    else:
                        tokens_to_add = list(RIMES_MATCHING_TOKENS.values())+list(RIMES_MATCHING_TOKENS.keys())#[]

                    for token_to_add in tokens_to_add:
                        self.subword_tokenizer.add_tokens([token_to_add])

                if 'add_NEs_in_charset' in params and params['add_NEs_in_charset']:
                    # Add the Exo-POPP NE tokens:
                    for token_to_add in MATCHING_NAMED_ENTITY_TOKENS_NO_HIERARCHY:
                        self.subword_tokenizer.add_tokens([token_to_add])

            self.subword_tokenizer.max_char_prediction = params['max_char_prediction'] if 'max_char_prediction' in params else 768

            self.vocab = self.subword_tokenizer.get_vocab()
            self.reversed_vocab = {val:key for key,val in self.vocab.items()}
            self.added_vocab = self.subword_tokenizer.get_added_vocab()
            self.reversed_added_vocab = {val:key for key,val in self.added_vocab.items()}

            self.subword_dict = list(self.vocab.keys())
            self.subword_dict = sorted(self.subword_dict,key=lambda x: self.vocab[x])
        elif 'use_subwords' in self.params['config']['constraints']:
            # use another tokenizer that the one used by BART
            self.subword_tokenizer = Tokenizer.from_file("basic/subwords/tokenizer-my-wiki.json")
            self.subword_dict = sorted(list(self.subword_tokenizer.get_vocab().keys()),key=lambda x: self.subword_tokenizer.get_vocab()[x])

    def __getitem__(self, idx):
        try:
            sample = copy.deepcopy(self.samples[idx])
        except Exception as e:
            print(e)
            print(idx)
            print('Samples lenght', len(self.samples))

        if not self.load_in_memory:
            sample["img"] = self.get_sample_img(idx)
            sample = apply_preprocessing(sample, self.params["config"]["preprocessings"])

        if (
            "synthetic_data" in self.params["config"]
            and self.params["config"]["synthetic_data"]
            and self.set_name == "train"
        ):
            sample = self.generate_synthetic_data(sample)
            if len(sample["img"].shape) == 2:
                sample["img"] = np.expand_dims(sample["img"], axis=2)

        try:
            sample["img"], sample["applied_da"] = self.apply_data_augmentation(sample["img"])
        except:
            sample["applied_da"] = []

        if "max_size" in self.params["config"] and self.params["config"]["max_size"]:
            max_ratio = max(
                sample["img"].shape[0] / self.params["config"]["max_size"]["max_height"],
                sample["img"].shape[1] / self.params["config"]["max_size"]["max_width"],
            )
            if max_ratio > 1:
                new_h, new_w = int(np.ceil(sample["img"].shape[0] / max_ratio)), int(
                    np.ceil(sample["img"].shape[1] / max_ratio)
                )
                sample["img"] = cv2.resize(sample["img"], (new_w, new_h))

        # Normalization if requested
        if "normalize" in self.params["config"] and self.params["config"]["normalize"]:
            sample["img"] = (sample["img"] - self.mean) / self.std

        sample["img_shape"] = sample["img"].shape
        sample["img_reduced_shape"] = np.ceil(sample["img_shape"] / self.reduce_dims_factor).astype(
            int
        )

        # Padding to handle CTC requirements
        if self.set_name == "train":
            max_label_len = 0
            height = 1
            ctc_padding = False
            if "CTC_line" in self.params["config"]["constraints"]:
                max_label_len = sample["label_len"]
                ctc_padding = True
            if ctc_padding and 2 * max_label_len + 1 > sample["img_reduced_shape"][1] * height:
                sample["img"] = pad_image_width_right(
                    sample["img"],
                    int(np.ceil((2 * max_label_len + 1) / height) * self.reduce_dims_factor[1]),
                    self.padding_value,
                )
                sample["img_shape"] = sample["img"].shape
                sample["img_reduced_shape"] = np.ceil(
                    sample["img_shape"] / self.reduce_dims_factor
                ).astype(int)
            sample["img_reduced_shape"] = [max(1, t) for t in sample["img_reduced_shape"]]

        sample["img_position"] = [[0, sample["img_shape"][0]], [0, sample["img_shape"][1]]]
        # Padding constraints to handle model needs
        if "padding" in self.params["config"] and self.params["config"]["padding"]:
            if self.set_name == "train" or not self.params["config"]["padding"]["train_only"]:
                min_pad = self.params["config"]["padding"]["min_pad"]
                max_pad = self.params["config"]["padding"]["max_pad"]
                pad_width = (
                    randint(min_pad, max_pad)
                    if min_pad is not None and max_pad is not None
                    else None
                )
                pad_height = (
                    randint(min_pad, max_pad)
                    if min_pad is not None and max_pad is not None
                    else None
                )

                sample["img"], sample["img_position"] = pad_image(
                    sample["img"],
                    padding_value=self.padding_value,
                    new_width=self.params["config"]["padding"]["min_width"],
                    new_height=self.params["config"]["padding"]["min_height"],
                    pad_width=pad_width,
                    pad_height=pad_height,
                    padding_mode=self.params["config"]["padding"]["mode"],
                    return_position=True,
                )
        sample["img_reduced_position"] = [
            np.ceil(p / factor).astype(int)
            for p, factor in zip(sample["img_position"], self.reduce_dims_factor[:2])
        ]
        return sample

    def get_charset(self):
        charset = set()
        for i in range(len(self.samples)):
            charset = charset.union(set(self.samples[i]["label"]))
        return charset

    def convert_labels(self):
        """
        Label str to token at character level
        """
        print("Labels preprocessing")
        for i in tqdm(range(len(self.samples))):
            self.samples[i] = self.convert_sample_labels(self.samples[i])

    def convert_sample_labels(self, sample):
        label = sample["label"]

        if 'use_subwords' in self.params["config"]["constraints"]:
            rpl_dict = self.params["config"].get('replace_dict_str_to_ind', None)

        if "use_subwords_bart" in self.params["config"]["constraints"] and 'IAM' in self.name:
            label = label.replace("ⓟ",'').replace('Ⓟ','')

        line_labels = label.split("\n")
        if self.params["config"].get('use_word_labels', True):
            if "remove_linebreaks" in self.params["config"]["constraints"]:
                full_label = label.replace("\n", " ").replace("  ", " ")
                word_labels = full_label.split(" ")
            else:
                full_label = label
                word_labels = label.replace("\n", " ").replace("  ", " ").split(" ")
        else:
            full_label = label
            word_labels = ['']

        sample["label"] = full_label
        if 'use_subwords' in self.params["config"]["constraints"]:
            is_bart = 'use_subwords_bart' in self.params["config"]["constraints"]
            sample["token_label"] = LM_str_to_ind_subwords(self.subword_tokenizer,full_label, is_bart, rpl_dict)
            if 'MULTI' in self.name:
                if 'tagging_mode' in self.params:
                    if 'NER' in sample['dataset'] or not "mono_start" in self.params["config"]["constraints"]:
                        sample["token_label"][0] = self.vocab['<s-' + sample['dataset'] + '>']
                    else:
                        sample["token_label"][0] = self.vocab['<s>']
                else:
                    if not "mono_start" in self.params["config"]["constraints"]:
                        # use one specific start token for each dataset
                        sample["token_label"][0] = self.vocab['<s-' + sample['dataset'] + '>']
                    else:
                        # use the same start token for all the datasets
                        sample["token_label"][0] = self.vocab['<s>']
            elif 'EXOPOPP' in self.name and 'tagging_mode' in self.params and sample.get('dataset', 'EXOPOPP') == 'EXOPOPP_NER' : # EXOPOPP NER
                sample["token_label"][0] = self.vocab['<s-EXOPOPP_NER>']
            elif 'IAM' in self.name and 'tagging_mode' in self.params and 'wiki_with_NEs_filtered' in self.params['config'].get('other_samples', ''): # IAM NER
                sample["token_label"][0] = self.vocab['<s-IAM_NER>']
        else:
            sample["token_label"] = LM_str_to_ind(self.charset, full_label)

        if "add_eot" in self.params["config"]["constraints"]:
            sample["token_label"].append(self.tokens["end"])
        sample["label_len"] = len(sample["token_label"])

        if "add_sot" in self.params["config"]["constraints"]:
            sample["token_label"].insert(0, self.tokens["start"])

        sample["line_label"] = line_labels
        if 'use_subwords' in self.params["config"]["constraints"]:
            is_bart = 'use_subwords_bart' in self.params["config"]["constraints"]
            if self.params["config"].get('use_line_labels', True):
                sample["token_line_label"] = [LM_str_to_ind_subwords(self.subword_tokenizer,l, is_bart, rpl_dict) for l in line_labels]
            else:
                sample["token_line_label"] = [[0]]
        else:
            sample["token_line_label"] = [LM_str_to_ind(self.charset, l) for l in line_labels]
        sample["line_label_len"] = [len(l) for l in line_labels]
        sample["nb_lines"] = len(line_labels)

        sample["word_label"] = word_labels

        if 'use_subwords' in self.params["config"]["constraints"]:
            is_bart = 'use_subwords_bart' in self.params["config"]["constraints"]
            if self.params["config"].get('use_word_labels', True):
                sample["token_word_label"] = [LM_str_to_ind_subwords(self.subword_tokenizer,l, is_bart, rpl_dict) for l in word_labels]
            else:
                sample["token_word_label"] = [[0]]

        else:
            sample["token_word_label"] = [LM_str_to_ind(self.charset, l) for l in word_labels]

        sample["word_label_len"] = [len(l) for l in word_labels]
        sample["nb_words"] = len(word_labels)
        return sample

    def generate_synthetic_data(self, sample):
        config = self.params["config"]["synthetic_data"]

        if not (config["init_proba"] == config["end_proba"] == 1):
            nb_samples = self.training_info["step"] * self.params["batch_size"]
            if config["start_scheduler_at_max_line"]:
                max_step = config["num_steps_proba"]
                current_step = max(
                    0,
                    min(
                        nb_samples
                        - config["curr_step"] * (config["max_nb_lines"] - config["min_nb_lines"]),
                        max_step,
                    ),
                )
                proba = (
                    config["init_proba"]
                    if self.get_syn_max_lines() < config["max_nb_lines"]
                    else config["proba_scheduler_function"](
                        config["init_proba"], config["end_proba"], current_step, max_step
                    )
                )
            else:
                proba = config["proba_scheduler_function"](
                    config["init_proba"],
                    config["end_proba"],
                    min(nb_samples, config["num_steps_proba"]),
                    config["num_steps_proba"],
                )
            if rand() > proba:
                return sample

        if "mode" in config and config["mode"] == "line_hw_to_printed":
            sample["img"] = self.generate_typed_text_line_image(sample["label"])
            return sample

        if "mode" in config and config["mode"] == "line_hw_to_printed_wiki":
            sample["label"] = ''
            while not sample["label"]:
                sample["label"] = generate_english_wiki_line_label(self, max_len=100)[0][0]
                sample['label'] = ' '.join(sample['label'].split()).replace('|','')
                for char in sample['label']:
                    if char not in self.charset:
                        sample['label'] = sample['label'].replace(char,'')

            sample = self.convert_sample_labels(sample)

            sample["img"] = self.generate_typed_text_line_image(sample["label"])
            return sample

        return self.generate_synthetic_page_sample()

    def get_syn_max_lines(self, suffix=""):
        max_nb_lines_name = "max_nb_lines" + suffix
        config = self.params["config"]["synthetic_data"]
        if 'curriculum' in config and config["curriculum"]:
            nb_samples = self.training_info["step"]*self.params["batch_size"]
            if config.get('step_function',None) is not None:
                nlines = config['step_function'](config['curr_step'], curr_step = nb_samples, training_info = self.training_info)
                return nlines
            else:
                max_nb_lines = min(config[max_nb_lines_name], (nb_samples-config["curr_start"]) // config["curr_step"]+1)
                return max(config["min_nb_lines"], max_nb_lines)
        return config[max_nb_lines_name] if max_nb_lines_name in config else 1

    def get_exopopp_padding_ratios(self):
        """
        Define randomly the parts of the image used by the document
        """
        pad_top_ratio = round(rand_uniform(0.07, 0.15), 3)
        pad_bot_ratio = round(rand_uniform(0.05, 0.2), 3)
        pad_left_left_page_ratio = round(rand_uniform(0.03, 0.3), 3)
        pad_right_left_page_ratio = round(rand_uniform(0.01, 0.05), 3)
        pad_left_right_page_ratio = round(rand_uniform(0.03, 0.12), 3)
        pad_right_right_page_ratio = round(rand_uniform(0.01, 0.3), 3)

        return (
            pad_top_ratio,
            pad_bot_ratio,
            pad_left_left_page_ratio,
            pad_right_left_page_ratio,
            pad_left_right_page_ratio,
            pad_right_right_page_ratio,
        )

    def generate_synthetic_page_sample(self):
        """
        Generates a complete synthetic document (single or double page)
        """
        config = self.params["config"]["synthetic_data"]
        max_lines = self.get_syn_max_lines()

        self.training_info['previous-nb-lines'] = max_lines

        crop = config["crop_curriculum"] and max_lines < config["max_nb_lines"]
        sample_id = self.synthetic_id
        self.synthetic_id += 1
        nb_pages = 2 if "double" in config["dataset_level"] else 1
        sample_name = f"synthetic_data_{sample_id}"

        # Dataset selection
        dataset_key = "MULTI_LINGUAL"
        multi_dataset_mode = dataset_key in self.params["datasets"]
        if multi_dataset_mode:
            selected_dataset = self._select_synthetic_dataset(config)
            sample_name = f"synthetic_{selected_dataset}_{sample_id}"
            bounds = self.index_datasets[selected_dataset]
            background_sample = copy.deepcopy(self.samples[randint(*bounds)])
        else:
            selected_dataset = list(self.params["datasets"].keys())[0]
            background_sample = copy.deepcopy(self.samples[randint(0, len(self))])

        backgrounds, pages = [], []

        h, w, c = self._get_image_dimensions(background_sample)
        page_width = w // 2 if nb_pages == 2 else w

        for i in range(nb_pages):
            nb_lines = randint(config["min_nb_lines"], max_lines + 1)
            background = self._create_blank_background(h, page_width, c)
            if i == 0 and nb_pages == 2:
                background[:, -2:, :] = 0  # visual separation

            side = ["left", "right"][i]
            backgrounds.append(background)

            page = self._generate_page_for_dataset(
                name=selected_dataset,
                bg=background,
                side=side,
                crop=crop,
                nb_lines=nb_lines,
                page_width=page_width,
                height=h,
                config=config
            )
            pages.append(page)

        sample = self._build_sample_output(sample_name, pages, nb_pages, backgrounds, selected_dataset)
        if 'use_subwords_bart' not in self.params["config"]["constraints"]:
            sample = self.normalize_label_with_charset(sample)
        return self.convert_sample_labels(sample)

    def _select_synthetic_dataset(self, config):
        if hasattr(self, 'synth_dataset') and self.synth_dataset:
            return self.synth_dataset

        datasets = config.get('datasets', ['IAM', 'RIMES', 'READ_2016'])
        if 'MULTI_synth_props' not in config:
            return datasets[randint(0, len(datasets))]

        weights = config['MULTI_synth_props']
        population = [d for d in datasets for _ in range(weights[d])]
        return population[randint(0, len(population))]

    def _get_image_dimensions(self, sample):
        if self.params['config']['load_in_memory']:
            h, w, c = sample["img"].shape
        else:
            w, h = Image.open(sample['path']).size
            c = 1 if self.params['config']['color_mode'] == 'L' else 3
        return h, w, c

    def _create_blank_background(self, h, w, c):
        dtype = 'uint8' if not self.params['config']['load_in_memory'] else self.samples[0]["img"].dtype
        return np.ones((h, w, c), dtype=dtype) * 255

    def _generate_page_for_dataset(self, name, bg, side, crop, nb_lines, page_width, height, config):
        if name == 'READ_2016':
            coords = {
                "left": int(0.15 * page_width) if side == "left" else int(0.05 * page_width),
                "right": int(0.95 * page_width) if side == "left" else int(0.85 * page_width),
                "top": int(0.05 * height),
                "bottom": int(0.85 * height)
            }
            return self.generate_synthetic_read2016_page(background=bg, coords=coords, side=side, crop=crop, nb_lines=nb_lines)

        elif name == 'RIMES':
            return self.generate_synthetic_rimes_page(background=bg, nb_lines=nb_lines, crop=crop)

        elif name in ['IAM', 'IAM_NER', 'EXOPOPP', 'EXOPOPP_HDS', 'EXOPOPP_global', 'EXOPOPP_man_tap']:
            ratios = self.get_exopopp_padding_ratios()
            coords = {
                "left": int(ratios[2] * page_width) if side == "left" else int(ratios[4] * page_width),
                "right": int((1 - ratios[3]) * page_width) if side == "left" else int((1 - ratios[5]) * page_width),
                "top": int(ratios[0] * height),
                "bottom": int((1 - ratios[1]) * height)
            }
            if "EXOPOPP" in name:
                return generate_synthetic_exopopp_page(self, bg, coords, side=side, crop=crop, nb_lines=nb_lines)
            return generate_synthetic_iam_mpopp_page(self, bg, coords, crop=crop, nb_lines=nb_lines, min_lines=config['min_nb_lines'], selected_dataset_name=name)

        raise NotImplementedError(f"Unsupported dataset: {name}")

    def _build_sample_output(self, name, pages, nb_pages, backgrounds, dataset_name=''):
        sample = {"name": name, "path": None}

        if nb_pages == 1:
            page = pages[0]
            sample.update({
                "img": page[0],
                "label_raw": page[1]["raw"],
                "label_begin": page[1]["begin"],
                "label_sem": page[1]["sem"],
                "label": page[1],
                "nb_cols": page[2]
            })
        else:
            h1, h2 = pages[0][0].shape[0], pages[1][0].shape[0]
            if h1 != h2:
                max_h = max(h1, h2)
                for i in range(2):
                    backgrounds[i] = backgrounds[i][:max_h]
                    backgrounds[i][:pages[i][0].shape[0]] = pages[i][0]
                    pages[i][0] = backgrounds[i]

            sample.update({
                "img": np.concatenate([pages[0][0], pages[1][0]], axis=1),
                "label_raw": pages[0][1]["raw"] + "\n" + pages[1][1]["raw"],
                "label_begin": pages[0][1]["begin"] + pages[1][1]["begin"],
                "label_sem": pages[0][1]["sem"] + pages[1][1]["sem"],
                "nb_cols": pages[0][2] + pages[1][2],
            })

        sample["label"] = sample["label_raw"]
        sample["unchanged_label"] = sample["label"]

        if "MULTI_LINGUAL" in self.params["datasets"].keys():
            if 'NER' in dataset_name:
                sample['dataset'] = dataset_name
            else:
                sample['dataset'] = dataset_name.split('_'[0])[0]
        return sample

    def normalize_label_with_charset(self, sample):
        convert_dict = {
            '#':'',
            '&':'',
            '=':'',
            "<":'',
            ">":'',
            '_':'-',
            '–':'-',
            "Â":'A',
            "È":'E',
            "É":'E',
            "Û":'U',
            "à":'a',
            "â":'a',
            "ä":'a',
            "ç":'c',
            "è":'e',
            "é":'e',
            "ê":'e',
            "ë":'e',
            "î":'i',
            "ï":'i',
            "ô":'o',
            "ö":'o',
            "ù":'u',
            "û":'u',
            "ü":'u',
            "ÿ":'y',
        }
        for c_to_rpl, rpl_c in convert_dict.items():
            if c_to_rpl in sample['label'] and c_to_rpl not in self.charset:
                sample['label'] = sample['label'].replace(c_to_rpl, rpl_c)
        return sample

    def generate_pg_text_from_complete_article(self, global_text, nb_lines, max_len, end_at_dot=False):
        """
        Generates a page text label from a Wikipedia article.

        Args:
            global_text (str): The complete article text.
            nb_lines (int): The number of lines to include in the page text.
            max_len (int): The maximum length of each line in the page text.
            min_len (int): The minimum length of each line in the page text.

        To lower the number of acces to the Wikipedia (which are costly), we sample a random article and we take sentences
        from this articles. The remaining sentences are sent back for the next synthetic data.

        Returns:
            tuple: A tuple containing the generated page text (list of strings) and the updated global text (str).
        """

        objective_nb_chars = max_len*nb_lines
        global_text = global_text.strip()

        if len(global_text) < objective_nb_chars:
            if 'MULTI' in self.name:
                corpus_sample = self.multi_samples['RIMES']
            else:
                corpus_sample = self.other_samples
            for sample in corpus_sample:
                if not global_text.endswith('.'):
                    global_text += '.'
                global_text += ' ' + '. '.join(sample)
                break

        total_nb_chars = len(global_text)
        sentences = global_text.split('. ')
        index_sentence = [i for i,m in enumerate(re.finditer('\. ', global_text)) if m.start() < total_nb_chars-objective_nb_chars]
        index_max_sentence = index_sentence[-1] if index_sentence else len(sentences)-1
        start_index = torch.randint(0, index_max_sentence+1, (1,))[0]
        new_sample_text = '. '.join(sentences[start_index:])

        sentences_begin = [0,] + [m.start() for i,m in enumerate(re.finditer('\. ', global_text))]
        index_end = bisect.bisect_left(sentences_begin, sentences_begin[start_index]+objective_nb_chars)
        global_text = ('. ').join(sentences[:start_index] + sentences[index_end:]) + '.'
        if global_text == '.':
            if 'MULTI' in self.name:
                corpus_sample = self.multi_samples['RIMES']
            else:
                corpus_sample = self.other_samples
            for sample in corpus_sample:
                global_text = '. '.join(sample)
                break

        new_text = add_newlines(new_sample_text, max_len=max_len)
        new_text = re.sub(',( ,)+', ', ', new_text)
        new_text = '\n'.join([iter_line for iter_line in new_text.split('\n') if len(iter_line.strip())>1])
        new_text = re.sub('  +', ' ', new_text)

        page_sample = new_text.split('\n')
        page_sample = [line.strip() for line in page_sample if line.strip()]

        pg_text = page_sample[:nb_lines]

        if end_at_dot and nb_lines > 2:
            full_text='\n'.join(page_sample[:nb_lines])
            if full_text.rfind('. ') > -1:
                full_text = full_text[:full_text.rfind('. ')+1]
                tmp_page_sample = full_text.split('\n')

                if len(tmp_page_sample) > 1:
                    pg_text = tmp_page_sample

        return pg_text, global_text

    def get_paragraph_rimes_wiki_opti(self, global_text, mode, nb_lines, max_len):
        """
        Generate a paragraph dictionary and updated global text based on the given parameters.

        Args:
            global_text (str): The complete article text.
            mode (str): The type of paragraph.
            nb_lines (int): The number of lines in the paragraph.
            max_len (int): The maximum length of the paragraph.

        Returns:
            dict: The paragraph dictionary containing the label and type.
            str: The updated global text.
        """

        if 'font_uniform_dict' not in self.params['config']['synthetic_data']['config']:
            if not 'normaliz-dict-path' in self.params['config']['synthetic_data']['config']:
                normaliz_font_path = 'normaliz-dict-fonts.json'
            else:
                normaliz_font_path = self.params['config']['synthetic_data']['config']['normaliz-dict-path']

            with open(normaliz_font_path, 'r') as f:
                self.params['config']['synthetic_data']['config']['font_uniform_dict'] = json.load(f)

        end_at_dot = False
        if mode == "Corps de texte" and self.params['config']['synthetic_data']['config'].get('rimes'):
            end_at_dot = self.params['config']['synthetic_data']['config']['rimes'].get('end_at_dot',False)

        pg = None
        bool_is_label_invalid = False
        while not pg or bool_is_label_invalid:
            pg, tmp_global_text = self.generate_pg_text_from_complete_article(global_text, nb_lines, max_len=max_len, end_at_dot=end_at_dot)

            bool_is_label_invalid = is_label_invalid(pg, self)
            if bool_is_label_invalid:
                if 'MULTI' in self.name:
                    corpus_sample = self.multi_samples['RIMES']
                else:
                    corpus_sample = self.other_samples
                for sample in corpus_sample:
                    global_text += '. '.join(sample)
                    break

        global_text = tmp_global_text

        pg_dict = {
            "label": "\n".join(pg),
            "type": mode,
        }

        return pg_dict, global_text

    def generate_rime_paragraphs_text(self, kept_modes, nb_lines, multi=False):
        """
        Generates paragraphs of text based on the given parameters.

        Args:
            kept_modes (list): A list of text blocks to keep.
            nb_lines (int): The number of lines to generate.
            multi (bool, optional): Whether to use multilingual samples. Defaults to False.

        Returns:
            dict: A dictionary containing the generated paragraphs for each text block.
        """

        paragraphs = {}
        if multi:
            tmp_samples = self.multi_samples['RIMES']
        else:
            tmp_samples = self.other_samples

        for new_sample in tmp_samples:
            wiki_article_text = '. '.join(new_sample)
            break

        total_tokens = 2
        tmp_used_lines = 0

        for mode in kept_modes:
            tmp_pg = self.get_paragraph_rimes(mode=mode, mix=True)['label']

            if mode == "Ouverture":
                tmp_pg = '\n'.join(tmp_pg.split('\n')[:2])
                tmp_lines = [tmp_line[:40] for tmp_line in tmp_pg.split('\n')]
                tmp_pg = '\n'.join(tmp_lines)

            tmp_nb_lines = len(tmp_pg.split('\n'))

            tmp_used_lines += tmp_nb_lines
            tmp_max_len = min(
                            max([len(elem) for elem in tmp_pg.split('\n')]),
                            150
            )
            new_pg, wiki_article_text = self.get_paragraph_rimes_wiki_opti(global_text=wiki_article_text, mode=mode, nb_lines=tmp_nb_lines, max_len=tmp_max_len)

            if hasattr(self, "subword_tokenizer"):
                pg_total_tokens = len(self.subword_tokenizer.encode(new_pg['label'])) # we add 2 for the start and end layout tokens of the paragraph
                if total_tokens + pg_total_tokens > self.subword_tokenizer.max_char_prediction:
                    nb_tokens_to_remove = total_tokens + pg_total_tokens-self.subword_tokenizer.max_char_prediction

                    if nb_tokens_to_remove >= pg_total_tokens-2: # break if we don't have space for the new paragraph
                        break

                    tmp_pg_bis = self.subword_tokenizer.decode(self.subword_tokenizer.encode(new_pg['label'])[1:-nb_tokens_to_remove+1])
                    if len(tmp_pg_bis) < 3: # break if the new paragraph is only made of layout tokens
                        break
                    new_pg['label'] = tmp_pg_bis
                    # we remove unwanted spaces and breaklines at the end of the paragraph:
                    if new_pg['label'].endswith(' \n '):
                        new_pg['label'] = new_pg['label'][:-3]
                    if new_pg['label'].endswith(' \n'):
                        new_pg['label'] = new_pg['label'][:-2]
                    if new_pg['label'].endswith(' '):
                        new_pg['label'] = new_pg['label'][:-1]

            paragraphs[mode] = new_pg
            if hasattr(self, "subword_tokenizer"):
                total_tokens = sum([len(self.subword_tokenizer.encode(pg['label'])) for pg in paragraphs.values()])+2

            if tmp_used_lines >= nb_lines:
                break

        return paragraphs

    def generate_synthetic_rimes_page(self, background, nb_lines=20, crop=False):
        """
        Generates a synthetic Rimes page using the provided background image and parameters.

        Args:
            background (numpy.ndarray): The background image for the page.
            nb_lines (int, optional): The number of lines to generate on the page. Defaults to 20.
            crop (bool, optional): Whether to crop the generated page. Defaults to False.

        Returns:
            The image of the synthetic page, the label of the page, and the number of columns.
        """
        max_pad_left_ratio = self.params['config']['synthetic_data']['config'].get('max_pad_left_ratio', 1)
        if "rimes_valid_fonts_list" in self.params["config"]["synthetic_data"]["config"]:
            rimes_hw_fonts = self.params["config"]["synthetic_data"]["config"]["rimes_valid_fonts_list"]['hw']
            rimes_printed_fonts = self.params["config"]["synthetic_data"]["config"]["rimes_valid_fonts_list"]['printed']
            hw_proba = self.params["config"]["synthetic_data"]["config"]["rimes_valid_fonts_list"]['hw-proba']
            use_both_font_types = True
        else:
            use_both_font_types = False
            if "rimes_valid_fonts" in self.params["config"]["synthetic_data"]["config"]:
                rimes_valid_fonts = self.params['config']["synthetic_data"]['config']["rimes_valid_fonts"]
            else:
                rimes_valid_fonts = self.params['config']["synthetic_data"]['config']["valid_fonts"]

        if 'MULTI' in self.name or (self.params['config'].get('layout_tokens_mode', '') == 'MULTI' and self.params['config'].get('multi_layout_monolingual', False)):
            matching_tokens = MULTI_MATCHING_TOKENS
            matching_tokens_str = MULTI_MATCHING_TOKENS_STR
        else:
            matching_tokens = RIMES_MATCHING_TOKENS
            matching_tokens_str = RIMES_MATCHING_TOKENS_STR
        h, w, _ = background.shape
        num_lines = list()

        if hasattr(self,'index_datasets'):
            selected_bounds = self.index_datasets['RIMES']
            self.tmp_samples = self.samples[selected_bounds[0]:selected_bounds[1]+1]
            multi = True
        else:
            self.tmp_samples = self.samples
            multi = False

        for s in self.tmp_samples:
            l = sum([len(p["label"].split("\n")) for p in s["paragraphs_label"]])
            num_lines.append(l)
        stats = self.stat_sem_rimes()
        ordered_modes = [
            "Corps de texte",
            "PS/PJ",
            "Ouverture",
            "Date, Lieu",
            "Coordonnées Expéditeur",
            "Coordonnées Destinataire",
        ]
        object_ref = ["Objet", "Reference"]
        random.shuffle(object_ref)
        ordered_modes = ordered_modes[:3] + object_ref + ordered_modes[3:]
        kept_modes = list()
        for mode in ordered_modes:
            if rand_uniform(0, 1) < stats[mode]:
                kept_modes.append(mode)

        if hasattr(self, "subword_tokenizer"):
            total_tokens = 2 # we add 2 for the start and end of the complete text

        paragraphs = self.generate_rime_paragraphs_text(kept_modes, multi=multi, nb_lines=nb_lines)

        # proba to set whole text region to uppercase
        if rand_uniform(0, 1) < 0.2 and "Corps de texte" in paragraphs:
            label_lines = paragraphs["Corps de texte"]["label"].split("\n")
            # for label_line in label_line:

            for k in range(len(label_lines)):
                if rand_uniform(0, 1) < 0.2:
                    label_lines[k] = label_lines[k].upper()
                    label_lines[k] = label_lines[k].replace("È", "E").replace("Ë", "E").replace("Û", "U").replace("Ù", "U").replace("Î", "I").replace("Ï", "I").replace("Â", "A").replace("Œ", "OE")

            paragraphs["Corps de texte"]["label"] = '\n'.join(label_lines)

        # proba to duplicate a line and place it randomly elsewhere, in a body region
        if rand_uniform(0, 1) < 0.1 and "Corps de texte" in paragraphs:
            labels = paragraphs["Corps de texte"]["label"].split("\n")
            duplicated_label = labels[randint(0, len(labels))]
            labels.insert(randint(0, len(labels)), duplicated_label)

            if hasattr(self, "subword_tokenizer"):
                pg_total_tokens = len(self.subword_tokenizer.encode(duplicated_label))+3 # we add 3 for the \n and the two spaces
                if total_tokens + pg_total_tokens > self.subword_tokenizer.max_char_prediction:
                    nb_tokens_to_remove = total_tokens + pg_total_tokens-self.subword_tokenizer.max_char_prediction

                    if not(nb_tokens_to_remove >= pg_total_tokens-2):
                        labels = self.subword_tokenizer.decode(self.subword_tokenizer.encode(labels)[1:-nb_tokens_to_remove+1])
                        if labels.endswith(' \n '):
                            labels = labels[:-3]
                        if labels.endswith(' \n'):
                            labels = labels[:-2]
                        if labels.endswith(' '):
                            labels = labels[:-1]

                        paragraphs["Corps de texte"]["label"] = "\n".join(labels)
                        total_tokens = sum([len(self.subword_tokenizer.encode(pg['label'])) for pg in paragraphs.values()])+2
            else:
                paragraphs["Corps de texte"]["label"] = "\n".join(labels)

        def add_newlines(text,every=100):
            """
            Add an ‘\n’ every 50 characters, shifting the ‘\n’ if necessary to avoid cutting off words
            """
            words = text.split()
            lines = []
            current_line = ""

            for word in words:
                if len(current_line + " " + word) > every:
                    lines.append(current_line.strip())
                    current_line = word
                else:
                    current_line += " " + word

            if current_line:
                lines.append(current_line.strip())
            return "\n".join(lines)

        for mode in paragraphs.keys():
            if paragraphs[mode]["label"].startswith("."):
                paragraphs[mode]["label"] = paragraphs[mode]["label"][1:]
            line_labels = paragraphs[mode]["label"].split("\n")
            if len(line_labels) == 0:
                print("ERROR")
            paragraphs[mode]["lines"] = list()
            if any([len(line_label)>100 for line_label in line_labels]):
                paragraphs[mode]["label"] = add_newlines(paragraphs[mode]["label"].replace('\n', ' '))
                line_labels = paragraphs[mode]["label"].split("\n")

            for line_label in line_labels:
                paragraphs[mode]["lines"].append(line_label)

        page_labels = {"raw": "", "begin": "", "sem": ""}
        top_limit = 0
        bottom_limit = h
        max_bottom_crop = 0
        min_top_crop = h
        has_opening = has_object = has_reference = False
        top_opening = top_object = top_reference = 0
        right_opening = right_object = right_reference = 0
        has_reference = False
        date_on_top = False
        date_alone = False

        kept_modes = [kept_mode for kept_mode in kept_modes if kept_mode in paragraphs.keys()]

        unique_font_path = None
        global_font_size_unique = None
        if self.params['config']['synthetic_data']['config'].get('rimes',None):
            if self.params['config']['synthetic_data']['config']['rimes'].get('one_font_per_pg',False):
                if "rimes_valid_fonts_list" in self.params["config"]["synthetic_data"]["config"]:
                    rimes_hw_fonts = self.params["config"]["synthetic_data"]["config"]["rimes_valid_fonts_list"]['hw']
                    rimes_printed_fonts = self.params["config"]["synthetic_data"]["config"]["rimes_valid_fonts_list"]['printed']
                    hw_proba = self.params["config"]["synthetic_data"]["config"]["rimes_valid_fonts_list"]['hw-proba']
                    valid_fonts_for_unique = rimes_hw_fonts if rand_uniform(0, 1) < hw_proba else rimes_printed_fonts
                else:
                    if "rimes_valid_fonts" in self.params["config"]["synthetic_data"]["config"]:
                        rimes_valid_fonts = self.params['config']["synthetic_data"]['config']["rimes_valid_fonts"]
                    else:
                        rimes_valid_fonts = self.params['config']["synthetic_data"]['config']["valid_fonts"]
                    valid_fonts_for_unique = self.params['config']['synthetic_data']['config']['valid_fonts']

                unique_font_path = valid_fonts_for_unique[randint(0, len(valid_fonts_for_unique))]

            if self.params['config']['synthetic_data']['config']['rimes'].get('one_size_per_pg',False):
                global_font_size_unique = randint(self.params['config']['synthetic_data']['config']["font_size_min"], self.params['config']['synthetic_data']['config']["font_size_max"] + 1)

        for mode in kept_modes:
            pg = paragraphs[mode]
            if len(pg["lines"]) > nb_lines:
                pg["lines"] = pg["lines"][:nb_lines]

            nb_lines -= len(pg["lines"])

            if unique_font_path:
                chosable_fonts = [unique_font_path]
            else:
                if use_both_font_types:
                    if rand_uniform(0, 1) < hw_proba:
                        chosable_fonts = rimes_hw_fonts
                    else:
                        chosable_fonts = rimes_printed_fonts
                else:
                    chosable_fonts = rimes_valid_fonts
            pg_image = self.generate_typed_text_paragraph_image(
                pg["lines"], padding_value=255, max_pad_left_ratio=max_pad_left_ratio, same_font_size=True, valid_fonts=chosable_fonts,  global_font_size_unique=global_font_size_unique
            )
            # proba to remove some interline spacing
            if rand_uniform(0, 1) < 0.1:
                try:
                    pg_image = apply_transform(pg_image, Tightening(color=255, remove_proba=0.75))
                except:
                    pass
            # proba to rotate text region
            if rand_uniform(0, 1) < 0.1:
                try:
                    pg_image = apply_transform(
                        pg_image, RandomRotation(degrees=10, expand=True, fill=255)
                    )
                except:
                    pass
            pg["added"] = True
            if mode == "Corps de texte":
                pg_image = resize_max(pg_image, max_height=int(0.5 * h), max_width=w)
                img_h, img_w = pg_image.shape[:2]
                min_top = int(0.4 * h)
                max_top = int(0.9 * h - img_h)
                top = randint(min_top, max_top + 1)
                left = randint(0, int(w - img_w) + 1)
                bottom_body = top + img_h
                top_body = top
                bottom_limit = min(top, bottom_limit)
            elif mode == "PS/PJ":
                pg_image = resize_max(pg_image, max_height=int(0.06 * h), max_width=int(0.9 * w))
                img_h, img_w = pg_image.shape[:2]
                min_top = bottom_body
                max_top = int(min(h - img_h, bottom_body + 0.15 * h))
                try:
                    top = randint(min_top, max_top + 1)
                except:
                    pg["added"] = False
                    break
                left = randint(0, int(w - img_w) + 1)
                bottom_limit = min(top, bottom_limit)
            elif mode == "Ouverture":
                pg_image = resize_max(pg_image, max_height=int(0.05 * h), max_width=int(0.9 * w))
                img_h, img_w = pg_image.shape[:2]
                min_top = int(top_body - 0.05 * h)
                max_top = top_body - img_h
                try:
                    top = randint(min_top, max_top + 1)
                except:
                    pg["added"] = False
                    break
                left = randint(0, min(int(0.15 * w), int(w - img_w)) + 1)
                has_opening = True
                top_opening = top
                right_opening = left + img_w
                bottom_limit = min(top, bottom_limit)
            elif mode == "Objet":
                pg_image = resize_max(pg_image, max_height=int(0.06 * h), max_width=int(0.9 * w))
                img_h, img_w = pg_image.shape[:2]
                max_top = (
                    top_reference - img_h
                    if has_reference
                    else top_opening - img_h
                    if has_opening
                    else top_body - img_h
                )
                min_top = int(max_top - 0.05 * h)
                try:
                    top = randint(min_top, max_top + 1)
                except:
                    pg["added"] = False
                    break
                left = randint(0, min(int(0.15 * w), int(w - img_w)) + 1)
                has_object = True
                top_object = top
                right_object = left + img_w
                bottom_limit = min(top, bottom_limit)
            elif mode == "Reference":
                pg_image = resize_max(pg_image, max_height=int(0.03 * h), max_width=int(0.9 * w))
                img_h, img_w = pg_image.shape[:2]
                max_top = (
                    top_object - img_h
                    if has_object
                    else top_opening - img_h
                    if has_opening
                    else top_body - img_h
                )
                min_top = int(max_top - 0.05 * h)
                try:
                    top = randint(min_top, max_top + 1)
                except:
                    pg["added"] = False
                    break
                left = randint(0, min(int(0.15 * w), int(w - img_w)) + 1)
                has_reference = True
                top_reference = top
                right_reference = left + img_w
                bottom_limit = min(top, bottom_limit)
            elif mode == "Date, Lieu":
                pg_image = resize_max(pg_image, max_height=int(0.05 * h), max_width=int(0.45 * w))
                # pg_image = resize_max(pg_image, max_height=int(0.03 * h), max_width=int(0.45 * w))
                img_h, img_w = pg_image.shape[:2]
                if h - max_bottom_crop - 10 > img_h and randint(0, 10) == 0:
                    top = randint(max_bottom_crop, h)
                    left = randint(0, w - img_w)
                else:
                    min_top = top_body - img_h
                    max_top = top_body - img_h
                    min_left = 0
                    # Check if there is enough place to put the date at the right side of opening, reference or object
                    if object_ref == ["Objet", "Reference"]:
                        have = [has_opening, has_object, has_reference]
                        rights = [right_opening, right_object, right_reference]
                        tops = [top_opening, top_object, top_reference]
                    else:
                        have = [has_opening, has_reference, has_object]
                        rights = [right_opening, right_reference, right_object]
                        tops = [top_opening, top_reference, top_object]
                    for right_r, top_r, has_r in zip(rights, tops, have):
                        if has_r:
                            if right_r + img_w >= 0.95 * w:
                                max_top = min(top_r - img_h, max_top)
                                min_left = 0
                            else:
                                min_left = max(min_left, right_r + 0.05 * w)
                                min_top = top_r - img_h if min_top == top_body - img_h else min_top
                    if min_left != 0 and randint(0, 5) == 0:
                        min_left = 0
                        for right_r, top_r, has_r in zip(rights, tops, have):
                            if has_r:
                                max_top = min(max_top, top_r - img_h)

                    max_left = max(min_left, w - img_w)

                    # No placement found at right-side of opening, reference or object
                    if min_left == 0:
                        # place on the top
                        if randint(0, 2) == 0:
                            min_top = 0
                            max_top = int(min(0.05 * h, max_top))
                            date_on_top = True
                        # place just before object/reference/opening
                        else:
                            min_top = int(max(0, max_top - 0.05 * h))
                            date_alone = True
                            max_left = min(max_left, int(0.1 * w))

                    min_top = min(min_top, max_top)
                    top = randint(min_top, max_top + 1)
                    left = randint(int(min_left), max_left + 1)
                    if date_on_top:
                        top_limit = max(top_limit, top + img_h)
                    else:
                        bottom_limit = min(top, bottom_limit)
                    date_right = left + img_w
                    date_bottom = top + img_h
            elif mode == "Coordonnées Expéditeur":
                max_height = min(0.25 * h, bottom_limit - top_limit)
                if max_height <= 0:
                    pg["added"] = False
                    print("ko", bottom_limit, top_limit)
                    break
                pg_image = resize_max(pg_image, max_height=int(max_height), max_width=int(0.45 * w))
                img_h, img_w = pg_image.shape[:2]
                top = randint(top_limit, bottom_limit - img_h + 1)
                left = randint(0, int(0.5 * w - img_w) + 1)
            elif mode == "Coordonnées Destinataire":
                if h - max_bottom_crop - 10 > 0.2 * h and randint(0, 10) == 0:
                    pg_image = resize_max(
                        pg_image, max_height=int(0.2 * h), max_width=int(0.45 * w)
                    )
                    img_h, img_w = pg_image.shape[:2]
                    top = randint(max_bottom_crop, h)
                    left = randint(0, w - img_w)
                else:
                    max_height = min(0.25 * h, bottom_limit - top_limit)
                    if max_height <= 0:
                        pg["added"] = False
                        print("ko", bottom_limit, top_limit)
                        break
                    pg_image = resize_max(
                        pg_image, max_height=int(max_height), max_width=int(0.45 * w)
                    )
                    img_h, img_w = pg_image.shape[:2]
                    if date_alone and w - date_right - img_w > 11:
                        top = randint(0, date_bottom - img_h + 1)
                        left = randint(max(int(0.5 * w), date_right + 10), w - img_w)
                    else:
                        top = randint(top_limit, bottom_limit - img_h + 1)
                        left = randint(int(0.5 * w), int(w - img_w) + 1)

            bottom = top + img_h
            right = left + img_w
            min_top_crop = min(top, min_top_crop)
            max_bottom_crop = max(bottom, max_bottom_crop)
            try:
                background[top:bottom, left:right, ...] = pg_image
            except:
                pg["added"] = False
                nb_lines = 0
            pg["coords"] = {"top": top, "bottom": bottom, "right": right, "left": left}

            if nb_lines <= 0:
                break

        paragraphs = {pg_mode: pg for pg_mode, pg in paragraphs.items() if 'coords' in pg}

        sorted_pg = order_text_regions_rimes(paragraphs.values())
        for pg in sorted_pg:
            if "added" in pg.keys() and pg["added"]:
                pg_label = "\n".join(pg["lines"])
                mode = pg["type"]
                begin_token = matching_tokens_str[mode]
                end_token = matching_tokens[begin_token]
                page_labels["raw"] += pg_label
                page_labels["begin"] += begin_token + pg_label
                page_labels["sem"] += begin_token + pg_label + end_token
        if crop:
            if min_top_crop > max_bottom_crop:
                print("KO - min > MAX")
            elif min_top_crop > h:
                print("KO - min > h")
            else:
                background = background[min_top_crop:max_bottom_crop]
        return [background, page_labels, 1]

    def stat_sem_rimes(self):
        """
        Calculate the semantic statistics for the Rimes dataset.

        Returns:
            dict: A dictionary containing the proportion of each text block type in the dataset.
        """
        try:
            return self.rimes_sem_stats
        except Exception as e:
            stats = dict()
            for sample in self.tmp_samples:
                for pg in sample["paragraphs_label"]:
                    mode = pg["type"]
                    if mode == "Coordonnées Expéditeur":
                        if len(pg["label"]) < 50 and "\n" not in pg["label"]:
                            mode = "Reference"
                    if mode not in stats.keys():
                        stats[mode] = 0
                    else:
                        stats[mode] += 1
            for key in stats:
                stats[key] = max(0.10, stats[key] / len(self.tmp_samples))
            self.rimes_sem_stats = stats
            return stats

    def get_paragraph_rimes(self, mode="Corps de texte", mix=False):
        """
        Retrieves a paragraph with the text block type from the dataset.

        Args:
            mode (str, optional): The text block type of the paragraph to retrieve. Defaults to "Corps de texte".
            mix (bool, optional): Whether to mix the paragraphs if the mode is "Corps de texte". Defaults to False.

        Returns:
            dict: The retrieved paragraph.
        """

        while True:
            sample = self.tmp_samples[randint(0, len(self.tmp_samples))]

            random.shuffle(sample["paragraphs_label"])
            for pg in sample["paragraphs_label"]:
                pg_mode = pg["type"]
                if pg_mode == "Coordonnées Expéditeur":
                    if len(pg["label"]) < 50 and "\n" not in pg["label"]:
                        pg_mode = "Reference"
                if mode == pg_mode:
                    if mode == "Corps de texte" and mix:
                        return self.get_mix_paragraph_rimes(
                            mode, min(5, len(pg["label"].split("\n")))
                        )
                    else:
                        return pg

    def get_mix_paragraph_rimes(self, mode="Corps de texte", num_lines=10):
        """
        Returns a mix of random lines from paragraphs of a specified mode.

        Parameters:
        - mode (str): The text block type of paragraphs to select lines from. Default is "Corps de texte".
        - num_lines (int): The number of lines to return. Default is 10.

        Returns:
        A dictionary with the following keys:
        - "label" (str): The concatenated lines.
        - "type" (str): The text block type of the selected paragraphs.
        """
        res = list()
        while len(res) != num_lines:
            sample = self.tmp_samples[randint(0, len(self.tmp_samples))]
            random.shuffle(sample["paragraphs_label"])
            for pg in sample["paragraphs_label"]:
                pg_mode = pg["type"]
                if pg_mode == "Coordonnées Expéditeur":
                    if len(pg["label"]) < 50 and "\n" not in pg["label"]:
                        pg_mode = "Reference"
                if mode == pg_mode:
                    lines = pg["label"].split("\n")
                    res.append(lines[randint(0, len(lines))])
                    break
        return {
            "label": "\n".join(res),
            "type": mode,
        }

    def get_paragraph_rimes_wiki(self, mode="Corps de texte", nb_lines=1, max_len=150, multi=False):
        """
        Generates a paragraph of text from the RIMES dataset.

        Args:
            mode (str, optional): The text block type to generate. Defaults to "Corps de texte".
            nb_lines (int, optional): The number of lines in the paragraph. Defaults to 1.
            max_len (int, optional): The maximum length of the lines. Defaults to 150.
            multi (bool, optional): Whether to generate multiple paragraphs. Defaults to False.

        Returns:
            dict: A dictionary containing the generated paragraph and its type.
        """

        if 'font_uniform_dict' not in self.params['config']['synthetic_data']['config']:
            if not 'normaliz-dict-path' in self.params['config']['synthetic_data']['config']:
                normaliz_font_path = 'normaliz-dict-fonts.json'
            else:
                normaliz_font_path = self.params['config']['synthetic_data']['config']['normaliz-dict-path']

            with open(normaliz_font_path, 'r') as f:
                self.params['config']['synthetic_data']['config']['font_uniform_dict'] = json.load(f)

        pg = None
        while not pg or is_label_invalid(pg, self):
            pg, _ = generate_page_text(self, nb_lines, max_len=max_len, multi=multi, dataset_name='RIMES')

        return {
            "label": "\n".join(pg),
            "type": mode,
        }

    def generate_synthetic_read2016_page(
        self, background, coords, side="left", nb_lines=20, crop=False
    ):
        """
        Generate a synthetic page for READ

        Args:
            background: empty image with the shape of a real sample
            coords: coordinates of the valid writing area
            side: 'left' or 'right'
            nb_lines: total number of lines in the document including page number, body and annotations
            crop : whether to crop under the last line
        Returns:
            [
                background: the same background but filled with the generated elements
                page_labels: contains the labels of the page ('raw', 'begin', 'sem')
                nb_cols: number of columns (for example: 2 in the case of a margin annotation and a body)
            ]
        """
        config = self.params["config"]["synthetic_data"]

        multi = True if 'multi_samples' in self.params['config'] else False

        two_column = False  # will be set to True if there is annotations in the generated document
        matching_token = READ_MATCHING_TOKENS
        page_labels = {
            "raw": "",  # no layout tokens
            "begin": "ⓟ",  # label with layout tokens of beginning
            "sem": "ⓟ",  # label with layout tokens
        }
        area_top = coords["top"]
        area_left = coords["left"]
        area_right = coords["right"]
        area_bottom = coords["bottom"]

        ratio = 999999
        # generate the line for the written page number ###############################
        if 'read_valid_fonts_list' in self.params['config']['synthetic_data']['config']:
            read_valid_fonts = self.params["config"]["synthetic_data"]["config"]["read_valid_fonts_list"]['wiki']
        else:
            read_valid_fonts = self.params['config']['synthetic_data']['config'].get('read_valid_fonts',self.params['config']['synthetic_data']['config']['valid_fonts'])
        num_page_text_label = str(randint(0, 1000))
        num_page_img = self.generate_typed_text_line_image(num_page_text_label, valid_fonts=read_valid_fonts)

        fixed_source_corpus = None
        unique_font_path = None
        unique_size = None
        if self.params['config']['synthetic_data']['config'].get('read',None):
            if self.params['config']['synthetic_data']['config']['read'].get('one_font_per_pg',False):
                if 'read_valid_fonts_list' in self.params['config']['synthetic_data']['config']:
                    read_valid_fonts_list = self.params["config"]["synthetic_data"]["config"]["read_valid_fonts_list"]
                    extended_fonts = {
                        'wiki_de':read_valid_fonts_list['wiki']
                    }
                    use_extended_fonts = True
                else:
                    use_extended_fonts = False

                if multi:
                    tmp_samples = self.multi_samples['READ_2016']
                else:
                    tmp_samples = self.other_samples

                for new_sample in tmp_samples:
                    if isinstance(new_sample,dict):
                        fixed_source_corpus = new_sample['source'][0]
                    else:
                        fixed_source_corpus = 'wiki_de'
                    break

                valid_fonts_for_unique = extended_fonts[fixed_source_corpus] if use_extended_fonts else read_valid_fonts
                unique_font_path = valid_fonts_for_unique[randint(0, len(valid_fonts_for_unique))]
            if self.params['config']['synthetic_data']['config']['read'].get('one_size_per_pg',False):
                if 'font_size_min' in self.params['config']['synthetic_data']['config']['read']:
                    unique_size = randint(self.params['config']['synthetic_data']['config']['read']["font_size_min"], self.params['config']['synthetic_data']['config']['read']["font_size_max"] + 1)
                else:
                    unique_size = randint(self.params['config']['synthetic_data']['config']["font_size_min"], self.params['config']['synthetic_data']['config']["font_size_max"] + 1)

        if side == "left":
            background[
                area_top : area_top + num_page_img.shape[0],
                area_left : area_left + num_page_img.shape[1],
            ] = num_page_img
        else:
            background[
                area_top : area_top + num_page_img.shape[0],
                area_right - num_page_img.shape[1] : area_right,
            ] = num_page_img
        # patch the page number image on the page, at the top left of the valid area or at the top right

        for key in ["sem", "begin"]:
            page_labels[key] += "ⓝ"  # "ⓝ" : page number
        for key in page_labels.keys():
            page_labels[key] += num_page_text_label
        page_labels["sem"] += matching_token["ⓝ"]
        #########################################################
        nb_lines -= 1
        area_top = (
            area_top + num_page_img.shape[0] + randint(1, 20)
        )  # the other elements of the page are necessarily below the number
        ratio_ann = rand_uniform(
            0.6, 0.7
        )  # define the ratio of the width taken by the body (1-ratio of the annotation)
        # the body takes between 60% and 70% of the page width

        # there can be several sections for one page (=a body linked with one or more margin annotations)
        total_tokens = 2
        global_labels = []
        while nb_lines > 0 and total_tokens < self.subword_tokenizer.max_char_prediction:
            nb_body_lines = randint(1, nb_lines + 1)
            # There cannot be more lines in the margin than in the paragraph.
            body_labels = list()
            body_imgs = list()  # list of line images of a body
            max_width_body = int(np.floor(ratio_ann * (area_right - area_left)))
            max_width_ann = area_right - area_left - max_width_body
            if 'read' in config['config'] and 'pixels_per_char' in config['config']['read']:
                pixels_per_char = config['config']['read']['pixels_per_char']
            else:
                pixels_per_char = 12 if 'pixels_per_char' not in config['config'] else config['config']['pixels_per_char']
            # add the lines of the body:
            # generate_page_text(self, 5, max_len=100, min_len=0)
            if ('other_samples' in self.params['config'] and ('wikipedia' in self.params['config']['other_samples'] or 'read' in self.params['config']['other_samples'])) or 'multi_samples' in self.params['config']:
                body_labels, body_imgs = self.generate_read_line_imgs(
                    nb_lines=nb_body_lines, field_type="body", max_len=max_width_body//pixels_per_char, multi=multi, total_tokens=total_tokens, unique_font_path=unique_font_path, fixed_source_corpus=fixed_source_corpus, unique_size=unique_size
                )
            else:
                while nb_body_lines > 0:
                    current_nb_lines = 1
                    label, img = self.get_printed_line_read_2016("body")

                    nb_body_lines -= current_nb_lines
                    body_labels.append(label)
                    body_imgs.append(img)

            nb_body_lines = len(body_labels)
            global_labels.extend(body_labels)
            max_ann_lines = min(nb_body_lines, nb_lines - nb_body_lines)

            if hasattr(self, "subword_tokenizer"):
                total_tokens = sum([len(self.subword_tokenizer.encode(pg_label)) for pg_label in global_labels])+2

            # add the lines of the annotation:
            nb_ann_lines = randint(0, min(6, max_ann_lines + 1))
            # for the page, there is between 0 and 6 lines for annotation (or less than 6 if max_ann_lines <=5)
            ann_labels = list()
            ann_imgs = list()

            if nb_ann_lines > 0 and total_tokens < self.subword_tokenizer.max_char_prediction:
                if (not 'other_samples' in self.params['config'] or not 'wikipedia' in self.params['config']['other_samples']) and not 'multi_samples' in self.params['config']:
                    while nb_ann_lines > 0:
                        current_nb_lines = 1
                        label, img = self.get_printed_line_read_2016("annotation")

                        nb_ann_lines -= current_nb_lines
                        ann_labels.append(label)
                        ann_imgs.append(img)
                else:
                    ann_labels, ann_imgs = self.generate_read_line_imgs(
                        nb_lines=nb_ann_lines, field_type="annotation", max_len=max_width_ann//pixels_per_char, multi=multi, total_tokens=total_tokens,
                        unique_font_path=unique_font_path, fixed_source_corpus=fixed_source_corpus, unique_size=unique_size
                    )

                nb_ann_lines = len(ann_labels)
                global_labels.extend(ann_labels)

                if hasattr(self, "subword_tokenizer"):
                    total_tokens = sum([len(self.subword_tokenizer.encode(pg_label)) for pg_label in global_labels])+2

            if self.params['config']['synthetic_data']['config'].get('read') and self.params['config']['synthetic_data']['config']['read'].get('one_size_per_pg',False) and self.params['config']['synthetic_data']['config']['read'].get('one_font_per_pg',False):
                for img_list, max_width in zip([body_imgs, ann_imgs], [max_width_body, max_width_ann]):
                    for i in range(len(img_list)):
                        # if the generated image is too wide, we resize it to fit the allowed width for this type of line
                        if img_list[i].shape[1] > max_width:
                            ratio = min(max_width / img_list[i].shape[1],ratio)

                if ratio < 1:
                    for img_list, max_width in zip([body_imgs, ann_imgs], [max_width_body, max_width_ann]):
                        for i in range(len(img_list)):
                            if img_list[i].shape[2] == 2:
                                channel_axis = None
                            else:
                                channel_axis = 2
                            img_list[i] = transform_skimage.rescale(img_list[i],ratio,3,anti_aliasing=True,preserve_range=True,channel_axis=channel_axis)
                            if len(img_list[i].shape) == 2:
                                img_list[i] = np.expand_dims(img_list[i], axis=2)
            else:
                for img_list, max_width in zip([body_imgs, ann_imgs], [max_width_body, max_width_ann]):
                    for i in range(len(img_list)):
                        # if the generated image is too wide, we resize it to fit the allowed width for this type of line
                        if img_list[i].shape[1] > max_width:
                            ratio = max_width / img_list[i].shape[1]

                            if img_list[i].shape[2] == 2:
                                channel_axis = None
                            else:
                                channel_axis = 2
                            img_list[i] = transform_skimage.rescale(img_list[i],ratio,3,anti_aliasing=True,preserve_range=True,channel_axis=channel_axis)

                            if len(img_list[i].shape) == 2:
                                img_list[i] = np.expand_dims(img_list[i], axis=2)

            body_top = area_top
            body_height = 0  # total height of the body
            i_body = 0  # number of lines added to the body
            # body_imgs : list of line images of the body
            for (label, img) in zip(body_labels, body_imgs):
                remaining_height = area_bottom - body_top
                if img.shape[0] > remaining_height:
                    nb_lines = 0
                    break  # there is no space left in the body to add more lines
                background[
                    body_top : body_top + img.shape[0],
                    area_left + max_width_ann : area_left + max_width_ann + img.shape[1],
                ] = img
                body_height += img.shape[0]
                body_top += img.shape[0]
                nb_lines -= 1
                i_body += 1

            # body_top: after that, body_top becomes the lowest point of the body
            #####################################################################################
            ann_height = int(
                np.sum([img.shape[0] for img in ann_imgs])
            )  # sum of the heights of every annotations
            ann_top = (
                area_top + randint(0, body_height - ann_height + 1)
                if ann_height < body_height
                else area_top
            )

            largest_ann = (
                max([a.shape[1] for a in ann_imgs]) if len(ann_imgs) > 0 else max_width_ann
            )
            # the widest annotation or by default the margin width
            pad_ann = (
                randint(0, max_width_ann - largest_ann + 1) if max_width_ann > largest_ann else 0
            )
            # pads to the left the void area of the annotation margin if any
            ann_label_blocks = [
                list(),
            ]  # there can be several annotations for one page
            i_ann = 0
            ann_height = 0
            for (label, img) in zip(ann_labels, ann_imgs):
                remaining_height = body_top - ann_top
                if img.shape[0] > remaining_height:
                    break
                background[
                    ann_top : ann_top + img.shape[0],
                    area_left + pad_ann : area_left + pad_ann + img.shape[1],
                ] = img
                ann_height += img.shape[0]
                ann_top += img.shape[0]
                nb_lines -= 1
                two_column = True
                ann_label_blocks[-1].append(ann_labels[i_ann])
                i_ann += 1
                if randint(0, 10) == 0:  # one chance in 10 of creating a new annotation block
                    ann_label_blocks.append(list())
                    ann_top += randint(0, max(15, body_top - ann_top - 20))  # ?

            if self.params['config']['synthetic_data']['config'].get('read') and self.params['config']['synthetic_data']['config']['read'].get('padding_bottom_ratio_max',False):
                area_top = area_top + max(ann_height, body_height) + randint(33, 133)
            else:
                area_top = area_top + max(ann_height, body_height) + randint(25, 100)
            # lowest written point in the document + a bottom margin

            ##############################################################
            ann_full_labels = {
                "raw": "",
                "begin": "",
                "sem": "",
            }
            for ann_label_block in ann_label_blocks:
                if len(ann_label_block) > 0:
                    for key in ["sem", "begin"]:
                        ann_full_labels[key] += "ⓐ"  # "ⓐ" : annotation
                    ann_full_labels["raw"] += "\n"
                    for key in ann_full_labels.keys():  # 'raw', 'begin', 'sem'
                        ann_full_labels[key] += "\n".join(ann_label_block)
                    ann_full_labels["sem"] += matching_token["ⓐ"]

            ###############################################
            body_full_labels = {
                "raw": "",
                "begin": "",
                "sem": "",
            }
            if i_body > 0:  # if the body is not empty
                for key in ["sem", "begin"]:
                    body_full_labels[key] += "ⓑ"
                body_full_labels["raw"] += "\n"
                for key in body_full_labels.keys():
                    body_full_labels[key] += "\n".join(body_labels[:i_body])
                    # [:i_body] : body labels can have more lines that there was space for in the body
                body_full_labels["sem"] += matching_token["ⓑ"]

            section_labels = dict()
            for key in ann_full_labels.keys():
                section_labels[key] = ann_full_labels[key] + body_full_labels[key]
            for key in section_labels.keys():
                if section_labels[key] != "":
                    if key in ["sem", "begin"]:
                        section_labels[key] = (
                            "ⓢ" + section_labels[key]
                        )
                    if key == "sem":
                        section_labels[key] = section_labels[key] + matching_token["ⓢ"]
            # add the labels of the generated section to the labels of the page:
            for key in page_labels.keys():
                page_labels[key] += section_labels[key]

        if crop:  # crop the area of the image under the last line (area_top was modified)
            background = background[:area_top]

        page_labels["sem"] += matching_token["ⓟ"]  # close the page

        for key in page_labels.keys():
            page_labels[key] = page_labels[key].strip()
            # remove leading and trailing spaces

        return [background, page_labels, 2 if two_column else 1]

    def get_n_consecutive_lines_read_2016(self, n=1, mode="body"):
        while True:
            sample = self.samples[randint(0, len(self))]
            paragraphs = list()
            for page in sample["pages_label"]:
                paragraphs.extend(page["paragraphs"])
                random.shuffle(paragraphs)
                for pg in paragraphs:
                    if (
                        (mode == "body" and pg["mode"] == "body")
                        or (mode == "ann" and pg["mode"] == "annotation")
                    ) and len(pg["lines"]) >= n:
                        line_idx = randint(0, len(pg["lines"]) - n + 1)
                        lines = pg["lines"][line_idx : line_idx + n]
                        label = "\n".join([l["text"] for l in lines])
                        top = min([l["top"] for l in lines])
                        bottom = max([l["bottom"] for l in lines])
                        left = min([l["left"] for l in lines])
                        right = max([l["right"] for l in lines])
                        img = sample["img"][top:bottom, left:right]
                        return label, img

    def get_printed_line_read_2016(self, mode="body"):
        """
        Select a random sample from annotation, then a random paragraph then a random line until the lenght of the label
        is (> 5 when 'mode' == 'body or < 15 when 'mode' == 'annotation' ) and the label is not a digit

        Args:
            mode: 'body' or 'annotation'
        """
        while True:
            sample = self.samples[randint(0, len(self))]
            for page in sample["pages_label"]:
                paragraphs = list()
                paragraphs.extend(page["paragraphs"])
                random.shuffle(paragraphs)
                for pg in paragraphs:
                    random.shuffle(pg["lines"])
                    for line in pg["lines"]:
                        if (mode == "body" and len(line["text"]) > 5) or (
                            mode == "annotation"
                            and len(line["text"]) < 15
                            and not line["text"].isdigit()
                        ):
                            label = line["text"]
                            img = self.generate_typed_text_line_image(label)
                            return label, img

    def get_printed_line_german_wikipedia(self, mode="body"):
        """
        Select a random sample from annotation, then a random paragraph then a random line until the lenght of the label
        is (> 5 when 'mode' == 'body or < 15 when 'mode' == 'annotation' ) and the label is not a digit

        Args:
            mode: 'body' or 'annotation'
        """
        while True:
            sample = self.other_samples[randint(0, len(self.other_samples))]

            if (mode == "body" and len(sample) > 5) or (
                mode == "annotation"
                and len(sample) < 15
                and not sample.isdigit()
            ):
                label = sample
                img = self.generate_typed_text_line_image(label)
                return label, img

    def generate_read_line_imgs(self, nb_lines, field_type='paragraphe', max_len=150, multi=False, total_tokens=0, unique_font_path=None, fixed_source_corpus=None, unique_size=None):
        """
        Generates images of text lines for READ based on the specified parameters.

        Args:
            nb_lines (int): The number of lines to generate.
            field_type (str, optional): The type of field for the lines. Defaults to 'paragraphe'.
            max_len (int, optional): The maximum length of each line. Defaults to 150.
            multi (bool, optional): Whether the dataset is multilingual.
            total_tokens (int, optional): The current total of tokens in the pages.
                Useful to avoid overflow of the subword tokenizer. Defaults to 0.

        Returns:
            tuple: A tuple containing the generated page text and a list of generated images.
        """

        imgs = []
        page_text = []
        use_extended_fonts = False

        if 'read_valid_fonts_list' in self.params['config']['synthetic_data']['config']:
            read_valid_fonts_list = self.params["config"]["synthetic_data"]["config"]["read_valid_fonts_list"]
            extended_fonts = {
                'wiki_de':read_valid_fonts_list['wiki']
            }
            use_extended_fonts = True

        if 'read_valid_fonts' in self.params['config']['synthetic_data']['config']:
            read_valid_fonts = self.params['config']['synthetic_data']['config']['read_valid_fonts']
        else:
            read_valid_fonts = self.params['config']['synthetic_data']['config']['valid_fonts']

        if 'font_uniform_dict' not in self.params['config']['synthetic_data']['config']:
            if not 'normaliz-dict-path' in self.params['config']['synthetic_data']['config']:
                normaliz_font_path = 'normaliz-dict-fonts.json'
            else:
                normaliz_font_path = self.params['config']['synthetic_data']['config']['normaliz-dict-path']

            with open(normaliz_font_path, 'r') as f:
                self.params['config']['synthetic_data']['config']['font_uniform_dict'] = json.load(f)

        end_at_dot = False
        if self.params['config']['synthetic_data']['config'].get('read'):
            end_at_dot = self.params['config']['synthetic_data']['config']['read'].get('end_at_dot',False)

        while not page_text:
            page_text, source_corpus = generate_page_text(self, nb_lines, max_len=max_len, multi=multi, dataset_name='READ_2016', corpus_source=fixed_source_corpus, end_at_dot=end_at_dot)

        if total_tokens and hasattr(self, "subword_tokenizer"):
            page_text = self.subword_tokenizer.decode(self.subword_tokenizer.encode('\n'.join(page_text))[1:-1][:self.subword_tokenizer.max_char_prediction-total_tokens])
            page_text = page_text.replace(' \n ', '\n').split('\n')

        for i in range(len(page_text)):
            if i < len(source_corpus):
                read_valid_fonts = extended_fonts[source_corpus[i]] if (use_extended_fonts and source_corpus) else read_valid_fonts

            img = self.generate_typed_text_line_image(page_text[i], field_type=field_type,alinea=False, valid_fonts=read_valid_fonts, unique_font=unique_font_path, unique_size=unique_size, dataset_name='READ_2016')
            imgs.append(img)
        return page_text, imgs

    def generate_typed_text_line_image(self, text, field_type="",alinea=False, draw_line=False, valid_fonts=None, unique_font=None, unique_size=None, dataset_name=''):
        return generate_typed_text_line_image(
            text, self.params["config"]["synthetic_data"]["config"], field_type=field_type,alinea=alinea, draw_line=draw_line, valid_fonts=valid_fonts, unique_font=unique_font, unique_size=unique_size, dataset_name=dataset_name
        )

    def generate_typed_text_paragraph_image(
            self, texts, padding_value=255, max_pad_left_ratio=0.1, same_font_size=False, valid_fonts=None, global_font_size_unique=None
        ):
        """
        Generates a paragraph image with typed text.

        Args:
            texts (list): List of texts to be included in the paragraph.
            padding_value (int, optional): Value to be used for padding. Defaults to 255.
            max_pad_left_ratio (float, optional): Maximum ratio for left padding. Defaults to 0.1.
            same_font_size (bool, optional): Flag to indicate if all texts should have the same font size. Defaults to False.
            valid_fonts (list, optional): List of valid font paths. Defaults to None.
            global_font_size_unique (int, optional): Global font size to be used for all texts. Defaults to None.

        Returns:
            numpy.ndarray: Paragraph image.
        """

        config = self.params["config"]["synthetic_data"]["config"]
        valid_fonts = config["valid_fonts"] if valid_fonts is None else valid_fonts

        if 'font_uniform_dict' not in config:
            if not 'normaliz-dict-path' in config:
                normaliz_font_path = 'normaliz-dict-fonts.json'
            else:
                normaliz_font_path = config['normaliz-dict-path']

            with open(normaliz_font_path, 'r') as f:
                config['font_uniform_dict'] = json.load(f)

        font_uniform_dict = config.get('font_uniform_dict',{})

        if same_font_size:
            images = list()
            txt_color = config["text_color_default"]
            bg_color = config["background_color_default"]
            if global_font_size_unique:
                font_size = global_font_size_unique
            else:
                font_size = randint(config["font_size_min"], config["font_size_max"] + 1)
            for text in texts:
                line_image = None
                nb_try = 0
                while nb_try < 10 and line_image is None:
                    text_height=0
                    while not text_height:
                        font_path = valid_fonts[randint(0, len(valid_fonts))]
                        font_name = Path(font_path).name
                        if font_name in font_uniform_dict:
                            tmp_font_size = int(font_size * font_uniform_dict[font_name]['norm_factor'])
                        else:
                            tmp_font_size = copy.copy(font_size)
                        fnt = ImageFont.truetype(font_path, tmp_font_size)
                        text_width, text_height = fnt.getsize(text)

                    padding_top = int(
                        rand_uniform(config["padding_top_ratio_min"], config["padding_top_ratio_max"])
                        * text_height
                    )
                    padding_bottom = int(
                        rand_uniform(
                            config["padding_bottom_ratio_min"], config["padding_bottom_ratio_max"]
                        )
                        * text_height
                    )
                    padding_left = int(
                        rand_uniform(config["padding_left_ratio_min"], config["padding_left_ratio_max"])
                        * text_width
                    )
                    padding_right = int(
                        rand_uniform(
                            config["padding_right_ratio_min"], config["padding_right_ratio_max"]
                        )
                        * text_width
                    )
                    padding = [padding_top, padding_bottom, padding_left, padding_right]
                    line_image = generate_typed_text_line_image_from_params(
                            text, fnt, bg_color, txt_color, config["color_mode"], padding
                    )
                    nb_try += 1

                if line_image is not None:
                    images.append(line_image)
        else:
            images = [self.generate_typed_text_line_image(t, unique_font=valid_fonts, unique_size=global_font_size_unique) for t in texts]

        max_width = max([img.shape[1] for img in images])

        padded_images = [
            pad_image_width_random(
                img, max_width, padding_value=padding_value, max_pad_left_ratio=max_pad_left_ratio
            )
            for img in images
        ]
        return np.concatenate(padded_images, axis=0)

    def preprocess_img(self, sample):
        if "normalize" in self.params["config"] and self.params["config"]["normalize"]:
            sample["img"] = (sample["img"] - self.mean) / self.std

        sample["img_shape"] = sample["img"].shape
        sample["img_reduced_shape"] = np.ceil(sample["img_shape"] / self.reduce_dims_factor).astype(
            int
        )

        return sample

class OCRCollateFunction:
    """
    Merge samples data to mini-batch data for OCR task
    """

    def __init__(self, config):
        self.img_padding_value = float(config["padding_value"])
        self.label_padding_value = config["padding_token"]
        self.config = config

    def __call__(self, batch_data):
        names = [batch_data[i]["name"] for i in range(len(batch_data))]
        ids = [
            int.from_bytes(batch_data[i]["name"].split("/")[-1].split(".")[0].encode(), "little")
            for i in range(len(batch_data))
        ]
        applied_da = [batch_data[i]["applied_da"] for i in range(len(batch_data))]

        labels = [batch_data[i]["token_label"] for i in range(len(batch_data))]
        labels = pad_sequences_1D(labels, padding_value=self.label_padding_value)
        labels = torch.tensor(labels).long()
        reverse_labels = [
            [
                batch_data[i]["token_label"][0],
            ]
            + batch_data[i]["token_label"][-2:0:-1]
            + [
                batch_data[i]["token_label"][-1],
            ]
            for i in range(len(batch_data))
        ]
        reverse_labels = pad_sequences_1D(reverse_labels, padding_value=self.label_padding_value)
        reverse_labels = torch.tensor(reverse_labels).long()
        labels_len = [batch_data[i]["label_len"] for i in range(len(batch_data))]

        raw_labels = [batch_data[i]["label"] for i in range(len(batch_data))]
        unchanged_labels = [batch_data[i]["unchanged_label"] for i in range(len(batch_data))]

        nb_cols = [batch_data[i]["nb_cols"] for i in range(len(batch_data))]
        nb_lines = [batch_data[i]["nb_lines"] for i in range(len(batch_data))]

        padding_mode = self.config["padding_mode"] if "padding_mode" in self.config else "br"
        imgs = [batch_data[i]["img"] for i in range(len(batch_data))]
        imgs_shape = [batch_data[i]["img_shape"] for i in range(len(batch_data))]
        imgs_reduced_shape = [batch_data[i]["img_reduced_shape"] for i in range(len(batch_data))]
        imgs_position = [batch_data[i]["img_position"] for i in range(len(batch_data))]
        imgs_reduced_position = [
            batch_data[i]["img_reduced_position"] for i in range(len(batch_data))
        ]
        imgs = pad_images(imgs, padding_value=self.img_padding_value, padding_mode=padding_mode)
        imgs = torch.tensor(imgs).float().permute(0, 3, 1, 2)
        formatted_batch_data = {
            "names": names,
            "ids": ids,
            "nb_lines": nb_lines,
            "nb_cols": nb_cols,
            "labels": labels,
            "reverse_labels": reverse_labels,
            "raw_labels": raw_labels,
            "unchanged_labels": unchanged_labels,
            "labels_len": labels_len,
            "imgs": imgs,
            "imgs_shape": imgs_shape,
            "imgs_reduced_shape": imgs_reduced_shape,
            "imgs_position": imgs_position,
            "imgs_reduced_position": imgs_reduced_position,
            "applied_da": applied_da,
        }

        return formatted_batch_data

def generate_typed_text_line_image(
    text, config, bg_color=(255, 255, 255), txt_color=(0, 0, 0), field_type="",alinea=False,draw_line=False, valid_fonts=None, unique_font=None, unique_size=None, dataset_name=''
):
    """
    Generate a typed text image from a string.

    Args:
        text (str): The input text to generate the image from.
        config (dict): The configuration dictionary containing various parameters for image generation.
        bg_color (tuple, optional): The background color of the image. Defaults to (255, 255, 255).
        txt_color (tuple, optional): The text color of the image. Defaults to (0, 0, 0).
        field_type (str, optional): The type of the field. Defaults to "".
        alinea (bool, optional): Whether to add an indentation to the text. Defaults to False.
        draw_line (bool, optional): Whether to draw a line below the text. Defaults to False.
        valid_fonts (list, optional): The list of valid fonts to choose from. Defaults to None.

    Returns:
        PIL.Image.Image: The generated typed text image.
    """
    if unique_font is None:
        valid_fonts = config["valid_fonts"] if valid_fonts is None else valid_fonts
    else:
        valid_fonts = [unique_font]

    if 'invisible_chars' in config:
        for inv_char in config['invisible_chars']:
            text = text.replace(inv_char, "")
    if text == "":
        text = " "
    if "text_color_default" in config:
        txt_color = config["text_color_default"]
    if "background_color_default" in config:
        bg_color = config["background_color_default"]

    if unique_size is not None:
        font_size = unique_size
    else:
        if "font_size_mins" in config and field_type in config["font_size_mins"]:
            font_size = randint(
                config["font_size_mins"][field_type], config["font_size_maxs"][field_type] + 1
            )
        else:
            font_size = randint(config["font_size_min"], config["font_size_max"] + 1)

    if 'font_uniform_dict' not in config:
        if not 'normaliz-dict-path' in config:
            normaliz_font_path = 'normaliz-dict-fonts.json'
        else:
            normaliz_font_path = config['normaliz-dict-path']

        with open(normaliz_font_path, 'r') as f:
            config['font_uniform_dict'] = json.load(f)

    font_uniform_dict = config.get('font_uniform_dict',{})

    text_height = 0
    while not text_height:
        font_path = valid_fonts[randint(0, len(valid_fonts))]
        font_name = Path(font_path).name
        if font_name in font_uniform_dict:
            font_size = int(font_size * font_uniform_dict[font_name]['norm_factor'])
        fnt = ImageFont.truetype(font_path, font_size)
        text_width, text_height = fnt.getsize(text)

    if dataset_name == 'READ_2016' and 'read' in config and 'padding_top_ratio_min' in config['read']:
        config_padding = config['read']
    else:
        config_padding = config

    padding_top = int(
        rand_uniform(config_padding["padding_top_ratio_min"], config_padding["padding_top_ratio_max"]) * text_height
    )
    padding_bottom = int(
        rand_uniform(config_padding["padding_bottom_ratio_min"], config_padding["padding_bottom_ratio_max"])
        * text_height
    )
    if field_type == 'marge_noms':
        padding_bottom=padding_bottom*2

    if alinea:
        padding_left=int(
            config["padding_left_ratio_max"]* text_width *rand_uniform(6,7)
        )
    else:
        padding_left = int(
        rand_uniform(config["padding_left_ratio_min"], config["padding_left_ratio_max"])
        * text_width
        )
    padding_right = int(
        rand_uniform(config["padding_right_ratio_min"], config["padding_right_ratio_max"])
        * text_width
    )
    padding = [padding_top, padding_bottom, padding_left, padding_right]
    return generate_typed_text_line_image_from_params(
        text, fnt, bg_color, txt_color, config["color_mode"], padding,draw_line=draw_line
    )

def generate_typed_text_line_image_from_params(
    text, font, bg_color, txt_color, color_mode, padding, draw_line=False
):
    """
    Generates a typed text line image from the given parameters.

    Args:
        text (str): The text to be displayed on the image.
        font (PIL.ImageFont.FreeTypeFont): The font to be used for the text.
        bg_color (str or tuple): The background color of the image.
        txt_color (str or tuple): The text color of the image.
        color_mode (str): The color mode of the image ("L" for grayscale, "RGB" for color).
        padding (tuple): The padding around the text (top, bottom, left, right).
        draw_line (bool, optional): Whether to draw a line on the image. Defaults to False.

    Returns:
        numpy.ndarray: The generated image as a NumPy array.
    """

    padding_top, padding_bottom, padding_left, padding_right = padding
    if draw_line:
        padding_top=max(padding_top*2,8)
    text_width, text_height = font.getsize(text)
    img_height = padding_top + padding_bottom + text_height
    img_width = padding_left + padding_right + text_width
    img = Image.new(color_mode, (img_width, img_height), color=bg_color)
    d = ImageDraw.Draw(img)
    if draw_line:
        x0=0
        x1=img_width-1
        y0=min(3,padding_top//2)
        y1=y0
        d.line([(x0, y0), (x1, y1)])
    d.text((padding_left, padding_bottom), text, font=font, fill=txt_color, spacing=0)
    if color_mode == "L":
        return np.expand_dims(np.array(img), axis=2)
    return np.array(img)

def get_valid_fonts(alphabet, fonts_path="Fonts"):
    """
    Retrieves a list of valid font paths based on the given alphabet.

    Args:
        alphabet (list): List of characters to check for in the fonts.
        fonts_path (str, optional): Path to the directory containing the fonts. Defaults to "Fonts".

    Returns:
        list: List of valid font paths.
    """

    if alphabet and "\n" in alphabet:
        alphabet.remove('\n')
    valid_fonts = list()
    for fold_detail in os.walk(fonts_path):
        if fold_detail[2]:
            for font_name in fold_detail[2]:
                if ".ttf" not in font_name or 'NotoEmoji' in font_name:
                    continue
                font_path = os.path.join(fold_detail[0], font_name)
                to_add = True
                if alphabet is not None:
                    for char in alphabet:
                        if not char_in_font(char, font_path):
                            to_add = False
                            break
                    if to_add:
                        valid_fonts.append(font_path)
                else:
                    valid_fonts.append(font_path)

    if 'Fonts/noto/NotoColorEmoji.ttf' in valid_fonts:
        valid_fonts.remove('Fonts/noto/NotoColorEmoji.ttf')
    return valid_fonts


def char_in_font(unicode_char, font_path):
    with TTFont(font_path) as font:
        for cmap in font["cmap"].tables:
            if cmap.isUnicode():
                if ord(unicode_char) in cmap.cmap:
                    return True
    return False
