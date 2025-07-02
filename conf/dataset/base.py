# This file is under a custom Research Usage Only (RUO) license.
# Please refer to the license file LICENSE for more details.
from basic.transforms import aug_config
from basic.scheduler import linear_scheduler

from OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
from OCR.ocr_dataset_manager import CustomXLMRobertaTokenizer

tokenizer = CustomXLMRobertaTokenizer.from_pretrained("basic/subwords/tokenizer-daniel")
tokenizer.__name__ = 'subword_tokenizer'

max_nb_lines = {
    "RIMES": 40,
    "READ_2016": 30,
    "MULTI_LINGUAL": 40,
    "IAM": 15,
    "IAM_NER": 15,
    "EXOPOPP_global":80,
    "custom_dataset": 40,
} # maximum number of lines for curriculum learning per dataset

def default_dataset_config_factory(dataset_name, dataset_level='page', dataset_variant='sem', multi=False):
    cfg = {
        "dataset_manager": OCRDatasetManager,
        "dataset_class": OCRDataset,
        "datasets": {
            dataset_name: "Datasets/formatted/{}_{}{}".format(dataset_name, dataset_level, dataset_variant),
        },
        "train": {
            "name": "{}-train".format(dataset_name),
            "datasets": [(dataset_name, "train"), ],
        },
        "valid": {
            "{}-valid".format(dataset_name): [(dataset_name, "valid"), ],
        },
        "worker_per_gpu": 4,  # Num of parallel processes per gpu for data loading
        "characters_vocab": None, # Characters vocabulary of the dataset, use to determine the compatible fonts in synthetic data generation. If None, the character set will be loaded from the labels file.
        "subword_tokenizer":tokenizer,
        "config": {
            "load_in_memory": True,  # Load all images in CPU memory
            "width_divisor": 8,  # Image width will be divided by 8
            "padding_value": 0,  # Image padding value
            "padding_token": -100,  # Label padding value
            "start_token": 0,  # Label padding value
            "end_token": 2,  # Label padding value
            "charset_mode": "seq2seq",  # add end-of-transcription ans start-of-transcription tokens to charset
            "constraints": ['use_subwords_bart','use_subwords'],
            "normalize": True,  # Normalize with mean and variance of training dataset
            "preprocessings": [
                {
                    # "type": "to_RGB",
                    "type": "to_grayscaled",
                },
                # {
                #   'type':'dpi',
                #   'source':300,
                #   'target':150
                # } # rescale the images on the fly with a factor of 0.5 (300/150)
            ],
            "augmentation": aug_config(0.9, 0.1),
            "synthetic_data": {
                "proba_scheduler_function": linear_scheduler,  # decrease proba rate linearly
                "start_scheduler_at_max_line": True,  # start decreasing proba only after curriculum reach max number of lines
                "dataset_level": dataset_level, # dataset level, page, paragraph or line
                "curriculum": True,  # use curriculum learning (slowly increase number of lines per synthetic samples)
                "crop_curriculum": True,  # during curriculum learning, crop images under the last text line
                "curr_start": 0,  # start curriculum at iteration
                "min_nb_lines": 3,# initial number of lines for curriculum learning
                "max_nb_lines": max_nb_lines[dataset_name],  # maximum number of lines for curriculum learning
                "padding_value": 255,
                # config for synthetic line generation
                "config": {
                    'pixels_per_char' : 14, # Minimum number of pixels in width per character
                    "fonts_path": "Fonts-everything", # path of the folder containing the font files
                    'normaliz-dict-path': 'font-normaliz-dict-everything.json', # path of the normalization dictionary for the font sizes
                    "background_color_default": 255, # default background color of the synthetic document images
                    "text_color_default": 0,
                    "font_size_min": 35, # minimum font size in the document
                    "font_size_max": 45, # maximum font size in the document
                    "color_mode": "L", # color mode of the synthetic document images, L for grayscale and RGB for color
                    # padding of the generated lines:
                    "padding_left_ratio_min": 0.00,
                    "padding_left_ratio_max": 0.06,
                    # left padding between 0 and 6% of the image width
                    "padding_right_ratio_min": 0.01,
                    "padding_right_ratio_max": 0.1,
                    # right padding between 1 and 10% of the image width
                    "padding_top_ratio_min": 0.02,
                    "padding_top_ratio_max": 0.1,
                    # top padding between 2 and 10% of the image height
                    "padding_bottom_ratio_min": 0.1,
                    "padding_bottom_ratio_max": 0.30,
                    # top padding between 10% and 30% of the image height
                    "invisible_chars": ['⭕', '蘋', '臧', '徠', '诛', '疸', '麾', '頜', '嗖', '頷', '勲', '麂', '掂', '砧', '👦', '裾', '📖', 'ǫ']
                },
            },
        },
    }

    if multi:
        del cfg["valid"]
    return cfg