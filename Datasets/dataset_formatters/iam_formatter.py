# This file is under a custom Research Usage Only (RUO) license.
# Please refer to the license file LICENSE for more details.
import json
import os
import pickle as pkl
from pathlib import Path
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.dirname(PARENT_DIR))
sys.path.append(os.path.dirname(os.path.dirname(PARENT_DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(PARENT_DIR))))
import numpy as np

from PIL import Image, ImageOps
from skimage import transform as transform_skimage

SEM_MATCHING_TOKENS = {
    "ⓟ": "Ⓟ",  # page
    # '':''
}

MULTI_MATCHING_TOKENS_STR = {
    'Ouverture': "ⓞ",  # opening
    'Corps de texte': "ⓒ",  # body
    'PS/PJ': "ⓔ",  # post scriptum
    'Coordonnées Expéditeur': "ⓕ",  # sender
    'Reference': "ⓕ",  # also counted as sender information
    'Objet': "ⓨ",  # why
    'Date, Lieu': "ⓦ",  # where, when
    'Coordonnées Destinataire': "ⓡ",  # recipient
}

MULTI_MATCHING_TOKENS = {
    # READ tokens (unchanged)
    "ⓑ": "Ⓑ",  # paragraph (body)
    "ⓐ": "Ⓐ",  # annotation
    "ⓟ": "Ⓟ",  # page
    "ⓝ": "Ⓝ",  # page number
    "ⓢ": "Ⓢ",  # section (=linked annotation + body)

    # RIMES tokens
    "ⓒ": "Ⓒ",  # body
    "ⓞ": "Ⓞ",# opening
    "ⓡ": "Ⓡ",# recipient
    "ⓕ": "Ⓕ",# sender and Reference
    "ⓦ": "Ⓦ",# where, when
    "ⓨ": "Ⓨ",# why
    "ⓔ": "Ⓔ", # post-scriptum

    # IAM tokens
    'ⓖ': 'Ⓖ',# IAM
}

EXO_POPP_MULTI_MATCHING_TOKENS = {
    'ⓗ': 'Ⓗ', # paragraph ExoPOPP
    'ⓙ': 'Ⓙ',# margin-names M-POPP
    "ⓘ": "Ⓘ",# margin-infos ExoPOPP
    'ⓜ': 'Ⓜ',# marriage act ExoPOPP
    'ⓚ': 'Ⓚ',# page ExoPOPP
}


def get_charset_espo(labels_dict):
    charset = set()
    for split_name, split_dict in labels_dict.items():
        for page_name, page_dict in split_dict.items():
            charset = charset.union(set(page_dict["text"]))

    return charset

emojis_to_name ={
    '⭕': 'DATE',
    '蘋': 'PER',
    '臧': 'GPE',
    '徠': 'LAW',
    '诛': 'ORG',
    '疸': 'PERCENT',
    '麾': 'MONEY',
    '頜': 'WORK_OF_ART',
    '嗖': 'CARDINAL',
    '頷': 'QUANTITY',
    '勲': 'NORP',
    '麂': 'LOCATION',
    '掂': 'TIME',
    '砧': 'EVENT',
    '👦': 'FAC',
    '裾': 'PRODUCT',
    '📖': 'ORDINAL',
    'ǫ': 'LANGUAGE',
    'O':''
}

ne_name_to_token = {
    'DATE': '⭕',
    'PER': '蘋',
    'GPE': '臧',
    'LAW': '徠',
    'ORG': '诛',
    'PERCENT': '疸',
    'MONEY': '麾',
    'WORK_OF_ART': '頜',
    'CARDINAL': '嗖',
    'QUANTITY': '頷',
    'NORP': '勲',
    'LOCATION': '麂',
    'TIME': '掂',
    'EVENT': '砧',
    'FAC': '👦',
    'PRODUCT': '裾',
    'ORDINAL': '📖',
    'LANGUAGE': 'ǫ',
    'O':''
}

MATCHING_NAMED_ENTITY_TOKENS = {
    '⭕': '',
    '蘋': '',
    '臧': '',
    '徠': '',
    '诛': '',
    '疸': '',
    '麾': '',
    '頜': '',
    '嗖': '',
    '頷': '',
    '勲': '',
    '麂': '',
    '掂': '',
    '砧': '',
    '👦': '',
    '裾': '',
    '📖': '',
    'ǫ': ''
}

def format_iam_images(imgs_path, bb_path, output_path, rescale=False, add_blank_margins=False):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    img_coords_dict = {}
    with open(bb_path, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                line = line.strip()
                columns = line.split(' ')
                line_id = columns[0]
                min_x, min_y, w, h = columns[4:8]
                min_x = int(min_x)
                min_y = int(min_y)
                max_x = min_x + int(w)
                max_y = min_y + int(h)

                img_name = '-'.join(line_id.split('-')[0:2])+'.png'
                if img_name not in img_coords_dict:
                    img_coords_dict[img_name] = {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}
                else:
                    img_coords_dict[img_name]['min_x'] = min(img_coords_dict[img_name]['min_x'], min_x)
                    img_coords_dict[img_name]['min_y'] = min(img_coords_dict[img_name]['min_y'], min_y)
                    img_coords_dict[img_name]['max_x'] = max(img_coords_dict[img_name]['max_x'], max_x)
                    img_coords_dict[img_name]['max_y'] = max(img_coords_dict[img_name]['max_y'], max_y)

    global_margin = 110 # margin in pixels that will be added to the image on the left, right and bottom
    left_margin = global_margin
    right_margin = global_margin
    top_margin = 50 # margin in pixels that will be added to the image on the top
    bottom_margin = global_margin

    for img_name, coords in img_coords_dict.items():
        img = Image.open(f'{imgs_path}/{img_name}')
        # (left, upper, right, lower)
        if add_blank_margins:
            # white bands of 100 pixels are added around the image
            border_size = 100
            pixels = list(img.getdata())
            # mean_value = sum(pixels) // len(pixels)
            mean_value = max(pixels)
            # Add a margin with the mean colour
            # we extract the paragraph image from the page image using the coordinates of the bounding box
            img = img.crop((coords['min_x'], coords['min_y'], coords['max_x'], coords['max_y']))
            img = ImageOps.expand(img, border=border_size, fill=mean_value)
        else:
            img = img.crop((max(coords['min_x']-left_margin,0), max(coords['min_y']-top_margin,0), min(coords['max_x']+right_margin,img.width), min(coords['max_y']+bottom_margin,img.height)))
        if rescale:
            channel_axis = None
            ratio = 1/2 # the image is rescaled at 150 dpi
            img = np.array(img)
            img = transform_skimage.rescale(img,ratio,3,anti_aliasing=True,preserve_range=True,channel_axis=channel_axis)
            img = Image.fromarray(img.astype(np.uint8))

        img.save(output_path + '/' + img_name)

def format_sentence_ne(sentence_id, sentences, named_entities_dict, ne_mode, previous_entity):
    words_list = []
    sentence = sentences[sentence_id]
    if int(sentence_id.split('-')[-1]):
        previous_sentence_id = '-'.join(sentence_id.split('-')[:2]) + '-' + str(int(sentence_id.split('-')[-1])-1).zfill(2)

    for i, word in enumerate(sentence.split('|')):
        word_id = f'{sentence_id}-{i:0>2d}'
        entity = named_entities_dict[word_id]
        if ne_mode == 'after':
            word = word + ne_name_to_token[entity]
        elif ne_mode == 'before':
            word = ne_name_to_token[entity] + word
        if ne_mode == 'both':
            if entity != previous_entity:
                if ne_name_to_token[previous_entity]:
                    if i:
                        words_list[i-1] += MATCHING_NAMED_ENTITY_TOKENS[ne_name_to_token[previous_entity]]
                    else:
                        sentences[previous_sentence_id] += MATCHING_NAMED_ENTITY_TOKENS[ne_name_to_token[previous_entity]]

                if ne_name_to_token[entity]:
                    word = ne_name_to_token[entity] + word
        words_list.append(word)

        previous_entity=entity
    sentence = ' '.join(words_list)
    sentences[sentence_id] = sentence
    return sentences, previous_entity

def format_labels_single_page_iam(
    labels_path, real_text_path, use_sem=True, use_nes=False, ne_mode='after', nb_entities=0, split_name='rwth'
):
    nb_cols = 1
    formatted_labels_dict = {"charset": [], "ground_truth": {}}

    word_split_dict = {subset : [] for subset in ['train', 'valid','test']}

    page_text_dict = {}
    with open(real_text_path, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                line = line.strip()
                columns = line.split(' ')
                line_id = columns[0]
                sentence = ' '.join(columns[8:])
                page_id = '-'.join(line_id.split('-')[0:2])
                if page_id in page_text_dict:
                    page_text_dict[page_id]['sentences'][line_id] = sentence
                else:
                    page_text_dict[page_id] = {'sentences':{line_id: sentence}}

    for subset in ['train', 'valid','test']:
        formatted_split_dict = {}
        page_label = ''
        named_entities_dict = {}

        with open(f'{labels_path}/iam_{subset}_{split_name}_{nb_entities}_all.txt', 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                word_id = line.split(' ')[0]
                word_split_dict[subset].append(word_id)

                if use_nes:
                    named_entity = line.split(' ')[1]
                    named_entities_dict[word_id] = named_entity

        for page_id, page_dict in page_text_dict.items():
            page_name = page_id + '.png'
            if not list(page_dict['sentences'].keys())[0]+'-00' in word_split_dict[subset]:
                continue
            formatted_split_dict[page_name] = {
                    "nb_cols": nb_cols,
                    "pages": [{"text": "", "nb_cols": nb_cols, "paragraphs": []}],
                }

            page_label = 'ⓟ' if use_sem else ''
            previous_entity = ''

            for line_id, sentence in page_dict['sentences'].items():
                if use_nes:
                    page_dict['sentences'], previous_entity = format_sentence_ne(line_id, page_dict['sentences'], named_entities_dict, ne_mode, previous_entity)

                page_dict['sentences'][line_id] = page_dict['sentences'][line_id].replace('|', ' ').replace(' ,', ',').replace(' .', '.').replace(' ;', ';').replace(' :', ':')

            page_label += '\n'.join(list(page_dict['sentences'].values()))
            page_label += 'Ⓟ' if use_sem else ''

            formatted_split_dict[page_name]["text"] = page_label
            formatted_split_dict[page_name]["pages"][0]["text"] = page_label

        formatted_labels_dict["ground_truth"][subset] = formatted_split_dict

    formatted_labels_dict["charset"] = get_charset_espo(formatted_labels_dict['ground_truth'])

    if use_sem:
        formatted_labels_dict["charset"] = formatted_labels_dict["charset"].union(
            ["ⓟ", "Ⓟ"]
        )
    if use_nes:
        formatted_labels_dict["charset"] = formatted_labels_dict["charset"].union(
            set([token for token in MATCHING_NAMED_ENTITY_TOKENS.values() if token]+[token for token in MATCHING_NAMED_ENTITY_TOKENS.keys() if token])
        )
    formatted_labels_dict["charset"] = sorted(list(formatted_labels_dict["charset"]))

    with open(Path(labels_path).joinpath("formatted-" + split_name + '-' + Path(labels_path).stem  + ".json"), "w") as f:
        json.dump(formatted_labels_dict, f, ensure_ascii=False)

    with open(
        Path(labels_path).joinpath("formatted-" + split_name + '-' + Path(labels_path).stem + ".pkl"), "wb"
    ) as f:
        pkl.dump(formatted_labels_dict, f)

if __name__ == "__main__":
    real_text_path = "Datasets/raw/IAM/lines.txt" # path of the file containing the text labels
    imgs_path = "Datasets/raw/IAM/images" # path of the folder containing the images

    mode = 'format-images'
    # format the images: crop the images using the bounding boxes, add small white margins so that the text is not at the edge of the image and rescale the images from 300 dpi to 150 dpi
    # mode = 'format-labels'
    # format the text labels using several parameters

    if mode == 'format-images':
        output_path = "Datasets/formatted/IAM_page_sem-150dpi-pg/flat" # path of the folder where the formatted images will be saved
        rescale=True # whether to rescale the images from 300 dpi to 150 dpi
        add_blank_margins=True # whether to add small white margins so that the text is not at the edge of the image
        format_iam_images(imgs_path, real_text_path, output_path, rescale=rescale, add_blank_margins=add_blank_margins)
    else:
        xml_folder_path = 'Datasets/raw/IAM-DB/18 entités'
        # use_sem = False
        use_sem = True
        # whether to use semantic tokens (ⓟ, Ⓟ)

        use_nes = True
        # use_nes = False
        # whether to use named entities

        # The IAM NER dataset exists in two versions:
        # - with 6 named entities (6 entities)
        # - with 18 named entities (18 entities)
        # The 6 named entities are: ['PER', 'LOC', 'ORG', 'GPE', 'MONEY', 'DATE']
        # The 18 named entities are: ['PER', 'LOC', 'ORG', 'GPE', 'MONEY', 'DATE', 'LAW', 'PERCENT', 'WORK_OF_ART', 'CARDINAL', 'QUANTITY', 'NORP', 'FAC', 'PRODUCT', 'ORDINAL', 'LANGUAGE']
        nb_entities = 18
        # nb_entities = 6

        # The following parameters define how the named entities are formatted in the text labels:
        # - 'after': the named entity token is added after the word
        # - 'before': the named entity token is added before the word
        ne_mode = 'after'
        # ne_mode = 'before'

        # The IAM NER dataset exists in two splits:
        # - 'rwth': a split designed for the htr task
        # - 'custom': a split designed for the ner task
        split_name = 'custom'
        # split_name = 'rwth'

        format_labels_single_page_iam(
        xml_folder_path, real_text_path, use_sem=use_sem, use_nes=use_nes, ne_mode=ne_mode, nb_entities=nb_entities, split_name=split_name)