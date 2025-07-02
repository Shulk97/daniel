# This file is under a custom Research Usage Only (RUO) license.
# Please refer to the license file LICENSE for more details.
import locale
import random

locale.setlocale(locale.LC_TIME,'')
import json
import re
import time
from copy import copy

import cv2
import numpy as np
import torch
from skimage import transform as transform_skimage

from basic.utils import rand_uniform, randint
from Datasets.dataset_formatters.exo_popp_formatter import \
    SEM_MATCHING_TOKENS as EXOPOPP_MATCHING_TOKENS
from Datasets.dataset_formatters.iam_formatter import GW_MATCHING_TOKENS

type_to_token = {"paragraphe": "ⓑ", "marge_noms": "ⓝ", "marge_info": "ⓘ","record": "ⓡ"}


def get_exopopp_line_image(mng, min_len=0, max_len=10, field_type="paragraphe", alinea=False, last_line=False):
    """
    Generates a labeled image of a line of text from a given document sample.

    Args:
        mng (object): The document sample manager object.
        min_len (int): The minimum length of the line of text (default is 0).
        max_len (int): The maximum length of the line of text (default is 10).
        field_type (str): The type of field to generate the line of text for (default is "paragraphe").
        alinea (bool): Whether to include an alinea in the generated line of text (default is False).
        last_line (bool): Whether the generated line of text is the last line of a paragraph (default is False).

    Returns:
        tuple: A tuple containing the label (text) and the generated image of the line of text.
    """

    while True:
        sample = mng.samples[randint(0, len(mng))]
        for page in sample["pages_label"]:
            paragraphs = [
                region_dict
                for region_dict in page["paragraphs"]
                if region_dict["mode"] == field_type
            ]
            random.shuffle(paragraphs)
            for pg in paragraphs:
                random.shuffle(pg["lines"])
                for line in pg["lines"]:
                    if len(line["text"]) > min_len and len(line["text"]) < max_len:
                        if (
                            field_type == "marge_noms"
                            and line["text"].lower().split("n°")[-1].isdigit()
                        ):
                            continue
                        label = line["text"]
                        if last_line: # the last line of a paragraph
                            nb_words = len(label.split(" ")) # number of words in the line
                            min_words = 1 if len(label.split(" ")[0])>=4 else 2 # we want at least 4 characters in the line or 2 words
                            new_nb_words = randint(min_words, nb_words+1)
                            label = ' '.join(label.split(" ")[:new_nb_words])
                        img = mng.generate_typed_text_line_image(label, field_type=field_type, alinea=alinea)
                        return label, img

def get_exopopp_rimes_line_image(mng, min_len=0, max_len=10, field_type='paragraphe', alinea=False, last_line=False):
    """
    Generates a line of text and its corresponding image for the Exopopp dataset
    but with the text labels from the RIMES dataset.

    Args:
        mng (object): The object that manages the dataset.
        min_len (int, optional): The minimum length of the line. Defaults to 0.
        max_len (int, optional): The maximum length of the line. Defaults to 10.
        field_type (str, optional): The type of field for the line. Defaults to 'paragraphe'.
        alinea (bool, optional): Whether to include an alinea in the line. Defaults to False.
        last_line (bool, optional): Whether the line is the last line of a paragraph. Defaults to False.

    Returns:
        tuple: A tuple containing the label of the line and its corresponding image.
    """

    while True:
        sample = mng.other_samples[randint(0, len(mng.other_samples))]
        for page in sample["pages_label"]:
            paragraphs = [
                region_dict
                for region_dict in page["paragraphs"]
            ]
            random.shuffle(paragraphs)
            for pg in paragraphs:
                lines = pg['label'].split('\n')
                random.shuffle(lines)
                for line in lines:
                    if len(line) > min_len and len(line) < max_len:
                        label = line.replace('¬', '')
                        if last_line:
                            nb_words = len(label.split(" ")) # number of words in the line
                            min_words = 1 if len(label.split(" ")[0])>=4 else 2 # we want at least 4 characters in the line or 2 words
                            new_nb_words = randint(min_words, nb_words+1)
                            label = ' '.join(label.split(" ")[:new_nb_words])
                        img = mng.generate_typed_text_line_image(label, field_type=field_type,alinea=alinea)
                        return label, img

def get_exopopp_wiki_line_image(mng, min_len=0, max_len=10, field_type='paragraphe', alinea=False, last_line=False, valid_fonts=None, unique_size=None):
    """
    Generates a labeled image of a text line using the given manager object.
    The text is taken from the Wikipedia corpus.

    Parameters:
    - mng: The manager object used for generating the image.
    - min_len: The minimum length of the line. Default is 0.
    - max_len: The maximum length of the line. Default is 10.
    - field_type: The type of the field. Default is 'paragraphe'.
    - alinea: A boolean indicating whether to include an alinea. Default is False.
    - last_line: A boolean indicating whether it is the last line. Default is False.

    Returns:
    - label: The label of the generated line.
    - img: The generated image of the line.
    """

    for elem in mng.other_samples:
        line = elem[0]
        if len(line) > min_len:
            label = line.replace('¬', '').replace('\n','')
            if len(label) > max_len:
                label = label[:max_len]
                last_space_index = label.rfind(' ')
                label = label[:last_space_index] if last_space_index > 0 else label
            elif last_line and len(label.split(" "))>2:
                nb_words = len(label.split(" ")) # number of words in the line
                min_words = 1 if len(label.split(" ")[0])>=12 else 3 # we want at least 4 characters in the line or 2 words
                new_nb_words = randint(min_words, nb_words+1)
                label = ' '.join(label.split(" ")[:new_nb_words])
            img = mng.generate_typed_text_line_image(label, field_type=field_type,alinea=alinea, valid_fonts=valid_fonts, unique_size=unique_size)
            return label, img

def generate_section_line_imgs(mng, nb_lines, field_type, max_len, min_len=0, valid_fonts=None, unique_size=None):
    """
    Generate synthetic section line images.

    Args:
        mng (object): The manager object.
        nb_lines (int): The number of lines to generate.
        field_type (str): The type of field.
        max_len (int): The maximum length of the line.
        min_len (int, optional): The minimum length of the line. Defaults to 0.

    Returns:
        tuple: A tuple containing the generated labels and images.
    """

    labels = []
    imgs = []
    alinea=False
    last_line = False
    sub_pg = randint(0, 10) == 1 # whether the paragraph is made of sub-paragraphs
    last_alinea=999
    while nb_lines > 0:
        if field_type =='paragraphe':
            if not labels and rand_uniform(0, 1) > 0.95: # 1/20 of proba
                alinea=True # the first line of a paragraph can have an alinea
                last_alinea=nb_lines
            else:
                alinea=False

            if nb_lines == 1:
                last_line = True
                # the last line of a paragraph can be shorter than the other lines
            elif sub_pg:
                # a sub paragraph begins with an alinea and ends with a short line
                if last_line:
                    alinea = True
                    last_alinea = nb_lines
                    last_line = False
                elif randint(0, 10) == 0 and last_alinea > nb_lines+1:
                    last_line=True

        if len(labels) > 3 and randint(0, 15) == 0:
            label = labels[randint(0, len(labels)-3)]
            img = mng.generate_typed_text_line_image(label, field_type=field_type,alinea=alinea, unique_size=unique_size)
            # chose a label already in the document to improve the attention robustness
        elif len(labels) > 3 and randint(0, 15) == 1:
            first_half_label = labels[randint(0, len(labels)-3)]
            first_half = ' '.join(first_half_label.split(' ')[:len(first_half_label.split(' '))//2])
            second_half_label = labels[randint(0, len(labels)-3)]
            second_half = ' '.join(second_half_label.split(' ')[len(second_half_label.split(' '))//2:])
            label = first_half + ' ' + second_half
            img = mng.generate_typed_text_line_image(label, field_type=field_type,alinea=alinea, unique_size=unique_size)
            # combines two lines already in the document to create a line with similar words than existing lines
        elif mng.other_samples:
            if isinstance(mng.other_samples,torch.utils.data.DataLoader):
                label, img = get_exopopp_wiki_line_image(mng, min_len=min_len, max_len=max_len, field_type=field_type, alinea=alinea, last_line=last_line, valid_fonts=valid_fonts, unique_size=unique_size)
            else:
                label, img = get_exopopp_rimes_line_image(mng, min_len=min_len, max_len=max_len, field_type=field_type, alinea=alinea, last_line=last_line)
        else:
            label, img = get_exopopp_line_image(mng,max_len=max_len, field_type=field_type, alinea=alinea, last_line=last_line)
        nb_lines -= 1
        labels.append(label)
        imgs.append(img)
    return labels, imgs

def is_label_invalid(label_text, mng, is_line=False):
    """
    Check if the label text is invalid based on certain criteria.
    This function avoid having text taking too many tokens to represent.
    Indeed, some special characters take a full token to be represented.

    Args:
        label_text (str): The label text to check.
        mng: The OCR dataset manager object.

    Returns:
        bool: True if the label text is invalid, False otherwise.
    """
    joined_text = ''.join(label_text)
    if is_line:
        is_content_invalid = any([
            joined_text.count('|') > 2,
            joined_text.count('(') > 10,
            joined_text.count('.') > 20,
            joined_text.count('-') > 20,
            joined_text.count(':') > 20,
            joined_text.count(',') > 20,
            joined_text.count('/') > 15,
            joined_text.count('\"') > 10,
            joined_text.count('\'') > 10,
        ])
    else:
        is_content_invalid = any([
            joined_text.count('|') > 2,
            joined_text.count('(') > 10,
            joined_text.count('.') > 40,
            joined_text.count('-') > 50,
            joined_text.count(':') > 40,
            joined_text.count(',') > 30,
            joined_text.count('/') > 15,
            joined_text.count('\"') > 50,
            joined_text.count('\'') > 30,
        ])
    if hasattr(mng, "subword_tokenizer"):
        is_content_invalid = is_content_invalid or (len(mng.subword_tokenizer.encode(joined_text)) > mng.subword_tokenizer.max_char_prediction)

    return is_content_invalid

def generate_section_line_imgs_subwords(mng, nb_lines, field_type='paragraphe', max_len=150, multi=False, valid_fonts=None, unique_size=None):
    """
    Generate synthetic section line images.

    Args:
        mng (object): The manager object.
        nb_lines (int): The number of lines to generate.
        field_type (str): The type of field.
        max_len (int): The maximum length of the line.

    Returns:
        tuple: A tuple containing the generated labels and images.
    """
    imgs = []
    alinea=False

    if field_type =='paragraphe':
        if rand_uniform(0, 1) > 0.95: # 1/20 of proba
            alinea = True

    page_text = []

    if 'font_uniform_dict' not in mng.params['config']['synthetic_data']['config']:
        if not 'normaliz-dict-path' in mng.params['config']['synthetic_data']['config']:
            normaliz_font_path = 'normaliz-dict-fonts.json'
        else:
            normaliz_font_path = mng.params['config']['synthetic_data']['config']['normaliz-dict-path']

        with open(normaliz_font_path, 'r') as f:
            mng.params['config']['synthetic_data']['config']['font_uniform_dict'] = json.load(f)

    patience = 0
    while not page_text or is_label_invalid(page_text, mng):
        if patience > 20 and len(mng.subword_tokenizer.encode('\n'.join(page_text))) > mng.subword_tokenizer.max_char_prediction:
            print('Consider increasing the max_char_prediction parameter')
            print('Text lenght:', len(mng.subword_tokenizer.encode('\n'.join(page_text))))
            print('Max char prediction:', mng.subword_tokenizer.max_char_prediction)
            page_text = mng.subword_tokenizer.decode(
                mng.subword_tokenizer.encode('\n'.join(page_text))[:mng.subword_tokenizer.max_char_prediction-10]
            ).split('\n')
            break
        try:
            page_text, _ = generate_page_text(mng, nb_lines, max_len=max_len, multi=multi, dataset_name='EXOPOPP_global')
        except Exception as e:
            print('Error:', e)
            print('nb lines', nb_lines, 'max len', max_len, 'multi', multi, 'longueur samples', len(mng.multi_samples['EXOPOPP_global']))
        patience += 1

    page_text = [normalize_label_text(mng, elem) for elem in page_text]

    if not valid_fonts:
        valid_fonts = mng.params["config"]["synthetic_data"]["config"].get("exopopp_valid_fonts",None)

    for i, _ in enumerate(page_text):
        img = mng.generate_typed_text_line_image(page_text[i], field_type=field_type,alinea=alinea, valid_fonts=valid_fonts, unique_size=unique_size)
        imgs.append(img)

    return page_text, imgs

def generate_iam_mpopp_line_imgs(mng, nb_lines, field_type='paragraphe', max_len=150, min_len=0):
    """
    Generates synthetic line images for Esposalles dataset.

    Args:
        mng (object): The manager object for handling dataset generation.
        nb_lines (int): The number of lines to generate.
        field_type (str, optional): The type of field for the lines. Defaults to 'paragraphe'.
        max_len (int, optional): The maximum length of the lines. Defaults to 150.
        min_len (int, optional): The minimum length of the lines. Defaults to 0.

    Returns:
        tuple: A tuple containing the generated labels and images.
    """

    labels = []
    imgs = []
    alinea=False
    total_lines = copy(nb_lines)
    last_line = False
    while nb_lines > 0:
        if field_type =='paragraphe':
            alinea = not labels and rand_uniform(0, 1) > 0.95 # the first line of a paragraph can have an alinea

            if nb_lines == 1 and total_lines>1:
                last_line = True
                # the last line of a paragraph can be shorter than the other lines

        if len(labels) > 3 and randint(0, 15) == 0:
            label = labels[randint(0, len(labels)-3)]
            img = mng.generate_typed_text_line_image(label, field_type=field_type,alinea=alinea)
            # chose a label already in the document to improve the attention robustness
        elif len(labels) > 3 and randint(0, 15) == 1:
            first_half_label = labels[randint(0, len(labels)-3)]
            first_half = ' '.join(first_half_label.split(' ')[:len(first_half_label.split(' '))//2])
            second_half_label = labels[randint(0, len(labels)-3)]
            second_half = ' '.join(second_half_label.split(' ')[len(second_half_label.split(' '))//2:])
            label = first_half + ' ' + second_half
            img = mng.generate_typed_text_line_image(label, field_type=field_type,alinea=alinea)
            # combines two lines already in the document to create a line with similar words than existing lines
        elif mng.other_samples:
            if isinstance(mng.other_samples,torch.utils.data.DataLoader):
                label, img = get_exopopp_wiki_line_image(mng, min_len=min_len, max_len=max_len, field_type=field_type, alinea=alinea, last_line=last_line)
            else:
                label, img = get_exopopp_rimes_line_image(mng, min_len=min_len, max_len=max_len, field_type=field_type, alinea=alinea, last_line=last_line)
        elif 'IAM' in mng.name:
            label, img = get_iam_line_image(mng,min_len=min_len, max_len=max_len, alinea=alinea, last_line=last_line, no_pg=True)
        else:
            label, img = get_iam_line_image(mng,min_len=min_len, max_len=max_len, alinea=alinea, last_line=last_line)
        nb_lines -= 1
        labels.append(label)
        imgs.append(img)
    return labels, imgs

def add_newlines(text, max_len=150):
    """
    Format a paragraph of text so that the line lenghts are max_len.

    Args:
        text (str): The input text.
        max_len (int, optional): The maximum length of each line. Defaults to 150.

    Returns:
        str: The text with newlines added.
    """

    words = text.split(' ')
    lines = []
    current_line = ''

    for word in words:
        if len(current_line) + len(word) > max_len:
            lines.append(current_line.strip())
            current_line = ''
        current_line += word + ' '

    lines.append(current_line.strip())

    return '\n'.join(lines)

def get_priorest_corpus(corpus_name_1, corpus_name_2):
    corpus_priority_dict = {
        'wiki_de': 0,
        'Unknown': -1
    }

    return corpus_name_1 if corpus_priority_dict[corpus_name_1] > corpus_priority_dict[corpus_name_2] else corpus_name_2

def generate_page_text(mng, nb_lines, max_len=150, multi=False, dataset_name=None, corpus_source=None, end_at_dot=False):
    """
    Generate text for a page in a synthetic document.

    Args:
        mng (object): The manager object containing the samples.
        nb_lines (int): The number of lines to generate for the page.
        max_len (int, optional): The maximum length of each line. Defaults to 150.
        multi (bool, optional): Flag indicating whether the training is multilingual. Defaults to False.
        dataset_name (str, optional): The name of the dataset to use for multilingual samples. Defaults to None.

    Returns:
        list: The generated lines for the page.
    """

    page_nb_lines = 0
    page_sample = []
    if multi:
        tmp_samples = mng.multi_samples[dataset_name]
    else:
        tmp_samples = mng.other_samples

    source_corpus = []

    for new_sample in tmp_samples:
        source = new_sample['source'][0] if isinstance(new_sample, dict) else 'wiki_de'

        if corpus_source and corpus_source != source:
            continue

        new_sample = new_sample['text'] if isinstance(new_sample, dict) else new_sample
        new_sample[0] = new_sample[0].replace('\n', ' ')
        new_sample[0] = re.sub(r' +', ' ', new_sample[0])

        if page_nb_lines < nb_lines:
            sentences = new_sample[0].strip().split('. ')

            if 'RIMES' in mng.name:
                total_nb_chars = len(new_sample[0].strip())
                objective_nb_chars = max_len*nb_lines
                index_sentence = [i for i,m in enumerate(re.finditer('\. ', new_sample[0].strip())) if m.start() < total_nb_chars-objective_nb_chars]
                index_max_sentence = index_sentence[-1] if index_sentence else len(sentences)-1
                start_index = torch.randint(0, index_max_sentence+1, (1,))[0]
                new_sample_text = '. '.join(sentences[start_index:])
            else:
                start_index = torch.randint(0, max(len(sentences)-nb_lines+1,1), (1,))[0]
                new_sample_text = '. '.join(sentences[start_index:start_index+nb_lines])

            if page_sample and page_sample[-1].endswith('.'):
                page_sample[-1] = page_sample[-1] + ' '

            new_sample_text = page_sample[-1]+new_sample_text if page_sample else new_sample_text
            # we concatenate the new text sample to the last line of the page if it exists

            new_text = add_newlines(new_sample_text, max_len=max_len)
            new_text = re.sub(r' +', ' ', new_text)

            if page_sample:
                page_sample = page_sample[:-1] + new_text.split('\n')
                priorest_corpus = get_priorest_corpus(source_corpus[-1], source)
                source_corpus = source_corpus[:-1] + [priorest_corpus] + [source]*(len(new_text.split('\n'))-1)
            else:
                page_sample.extend(new_text.split('\n'))
                source_corpus.extend([source]*len(new_text.split('\n')))

            page_sample = [line.strip() for line in page_sample if line.strip()]

            page_nb_lines = len(page_sample)

            if end_at_dot and nb_lines > 2 and page_nb_lines >= nb_lines:
                full_text='\n'.join(page_sample[:nb_lines])
                if full_text.rfind('. ') > -1:
                    if full_text[full_text.rfind('. ')-1].isdigit() and dataset_name == 'READ_2016':
                        if full_text[:full_text.rfind('. ')+1] == -1:
                            continue
                        full_text = full_text[:full_text.rfind('. ')+1]
                    full_text = full_text[:full_text.rfind('. ')+1]

                    page_sample = full_text.split('\n')

                    if len(page_sample) > 1:
                        return page_sample[:nb_lines], source_corpus
        else:
            return page_sample[:nb_lines], source_corpus



def generate_english_wiki_line_label(mng, max_len=150):
    """
    Generates a line label with text taken from English Wikipedia

    Args:
        mng: The manager object.
        max_len (int): The maximum length of the generated text. Default is 150.
        min_len (int): The minimum length of the generated text. Default is 10.

    Returns:
        tuple: A tuple containing the generated page text and a list of images.
    """

    imgs = []

    page_text = []
    while not page_text or is_label_invalid(page_text, mng, is_line=True):
        page_text, _ = generate_page_text(mng, 1, max_len=max_len)

    return page_text, imgs

def generate_iam_line_imgs(mng, nb_lines, field_type='paragraphe', max_len=150, multi=False, selected_dataset_name='IAM', unique_font_path=None, unique_size=None):
    imgs = []

    page_text = []

    if 'font_uniform_dict' not in mng.params['config']['synthetic_data']['config']:
        if not 'normaliz-dict-path' in mng.params['config']['synthetic_data']['config']:
            normaliz_font_path = 'normaliz-dict-fonts.json'
        else:
            normaliz_font_path = mng.params['config']['synthetic_data']['config']['normaliz-dict-path']

        with open(normaliz_font_path, 'r') as f:
            mng.params['config']['synthetic_data']['config']['font_uniform_dict'] = json.load(f)

    if "iam_valid_fonts_list" in mng.params["config"]["synthetic_data"]["config"]:
        hw_fonts = mng.params["config"]["synthetic_data"]["config"]["iam_valid_fonts_list"]['hw']
        printed_fonts = mng.params["config"]["synthetic_data"]["config"]["iam_valid_fonts_list"]['printed']
        hw_proba = mng.params["config"]["synthetic_data"]["config"]["iam_valid_fonts_list"]['hw-proba']
        valid_fonts = hw_fonts if rand_uniform(0, 1) < hw_proba else printed_fonts
    else:
        valid_fonts = None

    end_at_dot = False
    if mng.params['config']['synthetic_data']['config'].get('iam'):
        end_at_dot = mng.params['config']['synthetic_data']['config']['iam'].get('end_at_dot',False)

    while not page_text or is_label_invalid(page_text, mng):
        page_text, _ = generate_page_text(mng, nb_lines, max_len=max_len, multi=multi, dataset_name=selected_dataset_name, end_at_dot=end_at_dot)

    page_text = [normalize_label_text(mng, elem) for elem in page_text]

    for i, _ in enumerate(page_text):
        img = mng.generate_typed_text_line_image(page_text[i], field_type=field_type,alinea=False, valid_fonts=valid_fonts, unique_font=unique_font_path, unique_size=unique_size) # indiquer des fonts
        imgs.append(img)
    return page_text, imgs

def normalize_label_text(mng, line_text):
    """
    Normalize the label text by replacing specific characters and removing unsupported characters.

    Args:
        mng (object): The management object.
        line_text (str): The input line text.

    Returns:
        str: The normalized line text.
    """

    sample_charset = set(line_text)

    convert_dict = {
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

    if 'EXOPOPP_charset' in mng.params['config']:
        EXOPOPP_charset = mng.params['config']['EXOPOPP_charset']
        for char in EXOPOPP_charset:
            if char in convert_dict:
                del convert_dict[char]

    for char_to_rpl, rpl_char in convert_dict.items():
        line_text = line_text.replace(char_to_rpl, rpl_char)

    for char in sample_charset:
        if char not in mng.joined_charset and not char == ' ':
            line_text = line_text.replace(char, '')

    return line_text

def create_region_labels(matching_token, i_region, region_type, region_labels):
    """
    Create region labels based on the matching token, region index, region type, and region labels.

    Args:
        matching_token (dict): A dictionary containing matching tokens.
        i_region (int): The index of the current region.
        region_type (str): The type of the region.
        region_labels (list): A list of region labels.

    Returns:
        dict: A dictionary containing the full region labels.
    """
    full_labels = {
        "raw": "",
        "begin": "",
        "sem": "",
    }
    if i_region > 0:  # if the body is not empty
        for key in ["sem", "begin"]:
            full_labels[key] += type_to_token[region_type]
        if region_type != 'marge_noms':
            full_labels["raw"] += "\n"
        for key in full_labels:
            full_labels[key] += "\n".join(region_labels[:i_region])
        full_labels["sem"] += matching_token[type_to_token[region_type]]
    return full_labels

def create_iam_labels(i_region, region_type, region_labels):
    full_labels = {
        "raw": "",
        "begin": "",
        "sem": "",
    }
    if i_region > 0:  # if the body is not empty
        if region_type != 'marge_noms':
            full_labels["raw"] += "\n"
        for key in full_labels:
            full_labels[key] += "\n".join(region_labels[:i_region])
    return full_labels

def create_ann_labels(matching_token, ann_label_blocks):
    ann_full_labels = {
        "raw": "",
        "begin": "",
        "sem": "",
    }
    for ann_label_block in ann_label_blocks:
        if len(ann_label_block) > 0:
            for key in ["sem", "begin"]:
                ann_full_labels[key] += "ⓘ"
            ann_full_labels["raw"] += "\n"
            for key in ann_full_labels:  # 'raw', 'begin', 'sem'
                ann_full_labels[key] += "\n".join(ann_label_block)
            ann_full_labels["sem"] += matching_token["ⓘ"]
    return ann_full_labels

def create_section_labels(matching_token, marge_noms_full_labels, ann_full_labels, body_full_labels, record_key = "ⓜ", margin_first=False):
    section_labels = dict()
    for key in body_full_labels:
        if margin_first:
            section_labels[key] = (
                marge_noms_full_labels[key] + ann_full_labels[key] + body_full_labels[key]
            )
        else:
            section_labels[key] = (
                marge_noms_full_labels[key] + body_full_labels[key] + ann_full_labels[key]
            )
    for key in section_labels.keys():
        if section_labels[key] != "":
            if key in ["sem", "begin"]:
                section_labels[key] = (
                    record_key + section_labels[key]
                )
            if key == "sem":
                section_labels[key] = section_labels[key] + matching_token[record_key]
    return section_labels

def generate_date_text():
    if randint(0, 2) == 1:
        date = 'le '
    else:
        date = ''
    if randint(0, 2) == 1:
        return date + time.strftime("%d %B %Y",time.gmtime(random.randint(0, int(time.time()))))
    else:
        return date + time.strftime("%d %B",time.gmtime(random.randint(0, int(time.time()))))

def generate_marge_noms_imgs(mng, nb_marge_noms_lines, valid_fonts=None, unique_size=None):
    """
    Generate images and labels for the "marge_noms" field type.

    Args:
        mng (object): The object used for generating text line images.
        nb_marge_noms_lines (int): The number of lines to generate.

    Returns:
        tuple: A tuple containing two lists - marge_noms_labels and marge_noms_imgs.
            - marge_noms_labels (list): The generated labels for the "marge_noms" field type.
            - marge_noms_imgs (list): The generated images for the "marge_noms" field type.
    """

    if nb_marge_noms_lines == 0:
        return [], []

    has_num = False
    has_date = False

    rand_float = rand_uniform(0, 1)
    if rand_float < 0.9:
        has_num = True
        if rand_float < 0.4:
            marge_noms_label = str(randint(0, 1000))
        else:
            marge_noms_label = "N°" + str(randint(0, 1000))
        if rand_uniform(0, 1) > 0.9:
            marge_noms_label = "                     " + marge_noms_label
        num_page_img = mng.generate_typed_text_line_image(
            marge_noms_label, field_type="marge_noms", valid_fonts=valid_fonts
        )
        nb_marge_noms_lines -= 1

    if rand_uniform(0, 1) > 0.9:
        if rand_uniform(0, 1) > 0.9:
            draw_line=True
        else:
            draw_line=False
        has_date = True
        marge_noms_label_date = generate_date_text()
        date_img = mng.generate_typed_text_line_image(
            marge_noms_label_date, field_type="marge_noms", draw_line=draw_line, valid_fonts=valid_fonts, unique_size=unique_size
        )
        nb_marge_noms_lines -= 1

    if 'exopopp' in mng.params["config"]['synthetic_data']['config']:
        marge_noms_max_len = mng.params["config"]['synthetic_data']['config']['exopopp'].get('marge_noms_max_len', 30)
    else:
        marge_noms_max_len = 30

    if nb_marge_noms_lines > 0:
        if 'use_subwords_bart' in mng.params["config"]['constraints']:
            is_multi = 'MULTI' in mng.name
            marge_noms_labels, marge_noms_imgs = generate_section_line_imgs_subwords(
                mng, nb_lines=nb_marge_noms_lines, field_type="marge_noms", max_len=marge_noms_max_len, multi=is_multi, valid_fonts=valid_fonts, unique_size=unique_size
            )
        else:
            marge_noms_labels, marge_noms_imgs = generate_section_line_imgs(
                mng, nb_lines=nb_marge_noms_lines, field_type="marge_noms", max_len=marge_noms_max_len, valid_fonts=valid_fonts
            )
    else:
        marge_noms_labels = []
        marge_noms_imgs = []

    if has_date:
        marge_noms_labels.insert(len(marge_noms_labels), marge_noms_label_date)
        marge_noms_imgs.insert(len(marge_noms_labels), date_img)

    if has_num:
        if rand_uniform(0, 1) > 0.9:
            insert_index = len(marge_noms_labels)
        else:
            insert_index = 0
        marge_noms_labels.insert(insert_index, marge_noms_label)
        marge_noms_imgs.insert(insert_index, num_page_img)

    return marge_noms_labels, marge_noms_imgs

def resize_img_widths(img_list, max_width, same_resize=False):
    """
    Resize the widths of the images in the given list to fit within the specified maximum width.

    Parameters:
    img_list (list): A list of images to be resized.
    max_width (int): The maximum width allowed for the images.

    Returns:
    list: The resized image list.
    """
    ratio = None
    if same_resize:
        ratio = 99999
        for i in range(len(img_list)):
            # if the generated image is too wide, we resize it to fit the allowed width for this type of line
            if img_list[i].shape[1] > max_width:
                ratio = min(max_width / img_list[i].shape[1],ratio)

        if ratio < 1:
            for i in range(len(img_list)):
                if img_list[i].shape[2] == 2:
                    channel_axis = None
                else:
                    channel_axis = 2
                img_list[i] = transform_skimage.rescale(img_list[i],ratio,3,anti_aliasing=True,preserve_range=True,channel_axis=channel_axis)
                if len(img_list[i].shape) == 2:
                    img_list[i] = np.expand_dims(img_list[i], axis=2)
    else:
        for i in range(len(img_list)):
            # if the generated image is too wide, we resize it to fit the allowed width for this type of line
            if img_list[i].shape[1] > max_width:
                ratio = max_width / img_list[i].shape[1]
                new_h = int(np.floor(ratio * img_list[i].shape[0]))
                new_w = int(np.floor(ratio * img_list[i].shape[1]))
                img_list[i] = cv2.resize(
                    img_list[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR
                )
                if len(img_list[i].shape) == 2:
                    img_list[i] = np.expand_dims(img_list[i], axis=2)
    return img_list, ratio

def ensure_below_max_tokens(train_dataset, pg_text, total_tokens):
    do_break = False
    if hasattr(train_dataset, "subword_tokenizer"):
        pg_total_tokens = len(train_dataset.subword_tokenizer.encode(pg_text)) # we add 2 for the start and end layout tokens of the paragraph
        if total_tokens + pg_total_tokens > train_dataset.subword_tokenizer.max_char_prediction:
            nb_tokens_to_remove = total_tokens + pg_total_tokens + 2 -train_dataset.subword_tokenizer.max_char_prediction

            if nb_tokens_to_remove >= pg_total_tokens-12: # break if we don't have space for the new paragraph
                do_break = True

            tmp_pg_bis = train_dataset.subword_tokenizer.decode(train_dataset.subword_tokenizer.encode(pg_text)[1:-nb_tokens_to_remove+1])
            if len(tmp_pg_bis) < 3: # break if the new paragraph is only made of layout tokens
                do_break = True
            pg_text = tmp_pg_bis
            # we remove unwanted spaces and breaklines at the end of the paragraph:
            if pg_text.endswith(' \n '):
                pg_text = pg_text[:-3]
            if pg_text.endswith(' \n'):
                pg_text = pg_text[:-2]
            if pg_text.endswith(' '):
                pg_text = pg_text[:-1]

    return pg_text, do_break

def generate_synthetic_exopopp_page(train_dataset, background, coords, side="left", nb_lines=50, crop=False):
    """
    Generate a synthetic page for EXOPOPP

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
    exopopp_to_multi_dict = {
        "ⓟ":'ⓚ', # page ExoPOPP
        "ⓑ":'ⓗ', # paragraph ExoPOPP
        "ⓝ":'ⓙ', # margin-names M-POPP
        "ⓘ":'ⓘ', # margin-infos ExoPOPP
        "ⓜ":'ⓜ', # marriage act ExoPOPP

        "Ⓟ":'Ⓚ', # closing page ExoPOPP
        "Ⓑ":'Ⓗ', # closing paragraph ExoPOPP
        "Ⓝ":'Ⓙ', # closing margin-names ExoPOPP
        "Ⓘ":'Ⓘ', # closing margin infos ExoPOPP
        "Ⓜ":'Ⓜ', # closing marriage act ExoPOPP
    }
    total_tokens = 4 # useful when working with subwords
    # 2 for the start and end layout tokens of the sequence
    # and 2 for the start and end layout tokens of the page
    do_break = False
    do_break_pg = False
    do_break_ann = False
    config = train_dataset.params["config"]["synthetic_data"]

    if "rimes_valid_fonts_list" in config["config"]:
        rimes_hw_fonts = config["config"]["rimes_valid_fonts_list"]['hw']
        rimes_printed_fonts = config["config"]["rimes_valid_fonts_list"]['printed']
        hw_proba = config["config"]["rimes_valid_fonts_list"]['hw-proba']
        if rand_uniform(0, 1) < hw_proba:
            chosable_fonts = rimes_hw_fonts
        else:
            chosable_fonts = rimes_printed_fonts
    else:
        if "exopopp_valid_fonts" in config["config"]:
            chosable_fonts = config['config']["exopopp_valid_fonts"]
        else:
            chosable_fonts = config['config']["valid_fonts"]


    two_column = False  # will be set to True if there is annotations in the generated document
    matching_token = EXOPOPP_MATCHING_TOKENS
    page_labels = {
        "raw": "",  # no layout tokens
        "begin": "ⓟ",  # label with layout tokens of beginning
        "sem": "ⓟ",  # label with layout tokens
    }
    area_from_the_top = coords["top"]
    area_left = coords["left"]
    area_right = coords["right"]
    area_bottom = coords["bottom"]
    use_marge_noms = config['use_marge_noms'] if 'use_marge_noms' in config else True
    use_marge_infos = config['use_marge_infos'] if 'use_marge_infos' in config else True
    side = 'right'
    if 'side' in config and config['side'] == 'random' and randint(0, 2) == 0:
        side = 'left'

    unique_font_path = None
    unique_size_pg = None
    unique_size_marge_noms = None
    unique_size_pg_marge_infos = None
    if train_dataset.params['config']['synthetic_data']['config'].get('exopopp',None):
        if train_dataset.params['config']['synthetic_data']['config']['exopopp'].get('one_font_per_pg',False):
            if "rimes_valid_fonts_list" in train_dataset.params["config"]["synthetic_data"]["config"]:
                hw_fonts = train_dataset.params["config"]["synthetic_data"]["config"]["rimes_valid_fonts_list"]['hw']
                printed_fonts = train_dataset.params["config"]["synthetic_data"]["config"]["rimes_valid_fonts_list"]['printed']
                hw_proba = train_dataset.params["config"]["synthetic_data"]["config"]["rimes_valid_fonts_list"]['hw-proba']
                valid_fonts_for_unique = hw_fonts if rand_uniform(0, 1) < hw_proba else printed_fonts
            else:
                valid_fonts_for_unique = train_dataset.params['config']['synthetic_data']['config']['valid_fonts']

            unique_font_path = valid_fonts_for_unique[randint(0, len(valid_fonts_for_unique))]
            chosable_fonts = [unique_font_path]

        if train_dataset.params['config']['synthetic_data']['config']['exopopp'].get('one_size_per_pg',False):
            unique_size_marge_noms = randint(
                train_dataset.params['config']['synthetic_data']['config']["font_size_mins"]['marge_noms'],
                train_dataset.params['config']['synthetic_data']['config']["font_size_maxs"]['marge_noms'] + 1
            )
            unique_size_pg = randint(
                train_dataset.params['config']['synthetic_data']['config']["font_size_mins"]['paragraphe'],
                max(
                    min(
                        train_dataset.params['config']['synthetic_data']['config']["font_size_maxs"]['paragraphe'],
                        unique_size_marge_noms
                    ) + 1,
                    train_dataset.params['config']['synthetic_data']['config']["font_size_mins"]['paragraphe']+1
                )
            )
            unique_size_pg_marge_infos = randint(
                train_dataset.params['config']['synthetic_data']['config']["font_size_mins"]['marge_info'],
                max(
                    min(
                        train_dataset.params['config']['synthetic_data']['config']["font_size_maxs"]['marge_info'],
                        unique_size_pg
                    ) + 1,
                    train_dataset.params['config']['synthetic_data']['config']["font_size_mins"]['marge_info']+1
                )
            )

    #########################################################
    ratio_ann = rand_uniform(
        0.6, 0.75
    )  # define the ratio of the width taken by the body (1-ratio of the annotation)
    # the body takes between 60% and 70% of the page width
    remaining_height_body = area_bottom - area_from_the_top
    # there can be several sections for one page (=a body linked with one or more margin annotations)
    ind_pg = 0
    min_lines_license = 0
    while (nb_lines > min_lines_license and remaining_height_body > 0) and not (do_break or do_break_pg or do_break_ann):
        total_tokens += 2 # +2 for the start and end layout tokens of the marriage act
        min_lines_license=3
        min_lines_body = 3 if ind_pg else 1

        if (ind_pg == 0 and randint(0, 5) == 0) or not use_marge_noms:
            nb_marge_noms_lines = 0
        else:
            nb_marge_noms_lines = randint(
                1, max(min(nb_lines - min_lines_body, config["max_nb_lines_marge_noms"]), 2)
            )

        if randint(0, 5) != 0 and use_marge_infos:
            nb_ann_lines = randint(
                0,
                max(
                    min(
                        nb_lines - nb_marge_noms_lines - min_lines_body,
                        config["max_nb_lines_marge_info"],
                    ),
                    1,
                ),
            )
        else:
            nb_ann_lines = 0

        nb_body_lines = randint(min_lines_body, max(min(nb_lines + 1, config["max_nb_lines_paragraphe"]),min_lines_body+1))
        # add the lines of the marge_noms:
        marge_noms_labels, marge_noms_imgs = generate_marge_noms_imgs(train_dataset, nb_marge_noms_lines, valid_fonts=chosable_fonts, unique_size=unique_size_marge_noms)#,use_marge_noms)
        if 'use_subwords_bart' in train_dataset.params["config"]['constraints']:
            pg_text = '\n'.join(marge_noms_labels)
            pg_text, do_break = ensure_below_max_tokens(train_dataset, pg_text, total_tokens)
            if do_break:
                break
            total_tokens += len(train_dataset.subword_tokenizer.encode(pg_text)) + 2 # +2 for the start and end layout tokens
            marge_noms_labels = pg_text.split('\n')

        ################################################
        nb_pixels_per_char = train_dataset.params["config"]['synthetic_data']['config'].get('pixels_per_char',12)
        max_width_body = int(np.floor(ratio_ann * (area_right - area_left)))
        if 'use_subwords_bart' in train_dataset.params["config"]['constraints']:
            is_multi = 'MULTI' in train_dataset.name
            if not is_multi:
                print(train_dataset.name)
            body_labels, body_imgs = generate_section_line_imgs_subwords(
                train_dataset, nb_lines=nb_body_lines, field_type="paragraphe", max_len=max_width_body//nb_pixels_per_char, multi=is_multi, valid_fonts=chosable_fonts, unique_size=unique_size_pg
            )
            pg_text = '\n'.join(body_labels)
            pg_text, do_break_pg = ensure_below_max_tokens(train_dataset, pg_text, total_tokens)
            body_labels = pg_text.split('\n')
            if do_break_pg:
                break
            else:
                total_tokens += len(train_dataset.subword_tokenizer.encode(pg_text)) + 2 # +2 for the start and end layout tokens
        else:
            body_labels, body_imgs = generate_section_line_imgs(
                train_dataset, nb_lines=nb_body_lines, field_type="paragraphe", min_len=40, max_len=max_width_body//nb_pixels_per_char, valid_fonts=chosable_fonts, unique_size=unique_size_pg
            )

        ##############################################
        max_width_marg = area_right - area_left - max_width_body
        ##################################################
        # not less than 5 pixel per character
        if nb_ann_lines > 0:
            if 'use_subwords_bart' in train_dataset.params["config"]['constraints']:
                ann_labels, ann_imgs = generate_section_line_imgs_subwords(
                    train_dataset, nb_lines=nb_ann_lines, field_type="marge_info", max_len=max_width_marg//nb_pixels_per_char, multi=is_multi, valid_fonts=chosable_fonts, unique_size=unique_size_pg_marge_infos
                )
                pg_text = '\n'.join(ann_labels)
                pg_text, do_break_ann = ensure_below_max_tokens(train_dataset, pg_text, total_tokens)
                ann_labels = pg_text.split('\n')
                if not do_break_ann:
                    total_tokens += len(train_dataset.subword_tokenizer.encode(pg_text)) + 2 # +2 for the start and end layout tokens
            else:
                ann_labels, ann_imgs = generate_section_line_imgs(
                    train_dataset, nb_lines=nb_ann_lines, field_type="marge_info", max_len=max_width_marg//nb_pixels_per_char, valid_fonts=chosable_fonts
                )
        else:
            ann_labels, ann_imgs = [], []

        ## resize the generated images if needed: #################################
        same_resize = bool(unique_font_path and unique_size_pg)
        for img_list, max_width in zip(
            [marge_noms_imgs, body_imgs, ann_imgs],
            [max_width_marg, max_width_body, max_width_marg],
        ):
            img_list, _ = resize_img_widths(img_list, max_width, same_resize=same_resize)

        ###########################################################################################
        # apply the images from the body to the background starting from the top, until there are no more lines to place
        # to place or until there is no more room in the allocated space
        body_lowest_point = area_from_the_top
        body_height = 0  # total height of the body
        i_body = 0  # number of lines added to the body
        if side == 'left':
            body_leftest_point = area_left
        else:
            body_leftest_point = area_left + max_width_marg
        for img in body_imgs:
            remaining_height_body = area_bottom - body_lowest_point
            if img.shape[0] > remaining_height_body:
                nb_lines = 0
                break  # there is no space left in the body to add more lines

            background[
                body_lowest_point : body_lowest_point + img.shape[0],
                body_leftest_point : body_leftest_point + img.shape[1],
            ] = img
            body_height += img.shape[0]
            body_lowest_point += img.shape[0]
            nb_lines -= 1
            i_body += 1

        # body_top: after that, body_top becomes the lowest point of the body
        #####################################################################################
        marge_noms_lowest_point = area_from_the_top

        largest_margin_elem = (
            max([a.shape[1] for a in ann_imgs + marge_noms_imgs])
            if len(ann_imgs + marge_noms_imgs) > 0
            else max_width_marg
        )
        # the widest margin elem or by default the margin width
        pad_margin = (
            randint(0, max_width_marg - largest_margin_elem + 1)
            if max_width_marg > largest_margin_elem
            else 0
        )
        # pads to the left the void area of the annotation margin if any
        i_marge_noms = 0
        marge_noms_height = 0
        if side == 'left':
            margin_leftest_point = area_left + max_width_body
        else:
            margin_leftest_point = area_left
        for img in marge_noms_imgs:
            remaining_height_margin = body_lowest_point - marge_noms_lowest_point
            if img.shape[0] > remaining_height_margin:
                break
            background[
                marge_noms_lowest_point : marge_noms_lowest_point + img.shape[0],
                margin_leftest_point + pad_margin : margin_leftest_point + pad_margin + img.shape[1],
            ] = img
            marge_noms_height += img.shape[0]
            marge_noms_lowest_point += img.shape[0]
            nb_lines -= 1
            two_column = True
            i_marge_noms += 1

        marge_noms_lowest_point += randint(14, max(15, body_lowest_point - marge_noms_lowest_point - 20))
        ## add a random blank space below marge_noms
        #######################################################################################
        ann_height = int(
            np.sum([img.shape[0] for img in ann_imgs])
        )  # sum of the heights of every annotations
        remaining_ann_height = body_lowest_point - marge_noms_lowest_point - ann_height
        # remaining vertical space after having apply every line of this annotation
        ann_lowest_point = (
            marge_noms_lowest_point + randint(0, remaining_ann_height + 1)
            if remaining_ann_height > 0
            else marge_noms_lowest_point
        )
        # the annotation cannot extend beyond the main paragraph, but can be at a variable height below the name_margin

        ann_label_blocks = [
            list(),
        ]  # there can be several annotations for one page
        i_ann = 0
        ann_height = 0
        for img in ann_imgs:
            # ann_top goes lower and lower as annotation images are added
            remaining_height_margin = body_lowest_point - ann_lowest_point
            if img.shape[0] > remaining_height_margin:
                break
            background[
                ann_lowest_point : ann_lowest_point + img.shape[0],
                margin_leftest_point + pad_margin : margin_leftest_point + pad_margin + img.shape[1],
            ] = img
            ann_height += img.shape[0]
            ann_lowest_point += img.shape[0]
            nb_lines -= 1
            two_column = True
            ann_label_blocks[-1].append(ann_labels[i_ann])
            i_ann += 1
            if randint(0, 5) == 0:
                ann_label_blocks.append(list())
                ann_lowest_point += randint(14, max(15, body_lowest_point - ann_lowest_point))
                # space between the two annotation blocks, up to ‘remaining height-20’.

        area_from_the_top = max(ann_lowest_point, body_lowest_point) + randint(40, 150)
        # lowest written point in the document + a bottom margin

        ##############################################################
        # we successively add the different labels, starting with the annotations
        # then the paragraphs
        if do_break_pg or do_break_ann:
            ann_full_labels = {}
        else:
            ann_full_labels =  create_ann_labels(matching_token, ann_label_blocks)

        ###############################################
        marge_noms_full_labels = create_region_labels(matching_token, i_marge_noms, 'marge_noms', marge_noms_labels)

        ###############################################
        if do_break_pg:
            body_full_labels = {}
        else:
            body_full_labels = create_region_labels(matching_token, i_body, 'paragraphe', body_labels)

        margin_first = config.get('margin_first', False)
        section_labels = create_section_labels(matching_token, marge_noms_full_labels, ann_full_labels, body_full_labels, margin_first=margin_first)
        # add the labels of the generated section to the labels of the page:
        for key in page_labels:
            page_labels[key] += section_labels[key]

        ind_pg += 1

    if crop:  # crop the area of the image under the last line (area_top was modified)
        background = background[:area_from_the_top]

    page_labels["sem"] += matching_token["ⓟ"]  # close the page

    for key in page_labels:
        page_labels[key] = page_labels[key].strip()
        # remove leading and trailing spaces

    if 'MULTI' in train_dataset.params['config'].get('layout_tokens_mode', {}):
        for char_to_rpl, rpl_char in exopopp_to_multi_dict.items():
            for key in page_labels:
                page_labels[key] = page_labels[key].replace(char_to_rpl, rpl_char)

    return [background, page_labels, 2 if two_column else 1]

def generate_synthetic_iam_mpopp_page(mng, background, coords, nb_lines=50, crop=False, min_lines=1, selected_dataset_name='IAM'):
    """
    Generate a synthetic page for the IAM and M-POPP datasets

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
    config = mng.params["config"]["synthetic_data"]
    two_column = False  # will be set to True if there is annotations in the generated document
    if 'MULTI' in mng.name or (mng.params['config'].get('layout_tokens_mode', '') == 'MULTI' and mng.params['config'].get('multi_layout_monolingual', False)):
        matching_token = {
            'ⓖ':'Ⓖ',
            '':''
        }
        page_labels = {
            "raw": "",  # no layout tokens
            "begin": "ⓖ",  # label with layout tokens of beginning
            "sem": "ⓖ",  # label with layout tokens
        }  # "ⓟ": page
    else:
        matching_token = GW_MATCHING_TOKENS
        page_labels = {
            "raw": "",  # no layout tokens
            "begin": "ⓟ",  # label with layout tokens of beginning
            "sem": "ⓟ",  # label with layout tokens
        }  # "ⓟ": page
    area_from_the_top = coords["top"]
    area_left = coords["left"]
    area_right = coords["right"]
    area_bottom = coords["bottom"]

    #########################################################
    remaining_height_body = area_bottom - area_from_the_top
    # there can be several sections for one page (=a body linked with one or more margin annotations)
    ind_pg = 0
    min_lines_license = 0

    unique_font_path = None
    unique_size = None
    if mng.params['config']['synthetic_data']['config'].get('iam',None):
        if mng.params['config']['synthetic_data']['config']['iam'].get('one_font_per_pg',False):
            if "iam_valid_fonts_list" in mng.params["config"]["synthetic_data"]["config"]:
                hw_fonts = mng.params["config"]["synthetic_data"]["config"]["iam_valid_fonts_list"]['hw']
                printed_fonts = mng.params["config"]["synthetic_data"]["config"]["iam_valid_fonts_list"]['printed']
                hw_proba = mng.params["config"]["synthetic_data"]["config"]["iam_valid_fonts_list"]['hw-proba']
                valid_fonts_for_unique = hw_fonts if rand_uniform(0, 1) < hw_proba else printed_fonts
            else:
                valid_fonts_for_unique = mng.params['config']['synthetic_data']['config']['valid_fonts']

            unique_font_path = valid_fonts_for_unique[randint(0, len(valid_fonts_for_unique))]
        if mng.params['config']['synthetic_data']['config']['iam'].get('one_size_per_pg',False):
            unique_size = randint(mng.params['config']['synthetic_data']['config']["font_size_min"], mng.params['config']['synthetic_data']['config']["font_size_max"] + 1)

    while nb_lines > min_lines_license and remaining_height_body > 0:
        if selected_dataset_name in ['IAM','IAM_NER'] and mng.params['config']['synthetic_data']['config'].get('iam',None) and mng.params['config']['synthetic_data']['config']['iam'].get('single_pg',False):
            nb_body_lines = nb_lines
        else:
            min_lines_license=2
            min_lines_body = 2 if ind_pg else max(1,min_lines)

            nb_body_lines = randint(min_lines_body,
                max(
                    min(
                        nb_lines + 1,
                        config["max_nb_lines_paragraphe"]
                    ),
                    min_lines_body+1)
            )
        # add the lines of the marge_noms:
        ################################################
        max_width_body = int(np.floor(area_right - area_left))
        pixels_per_char = 12 if 'pixels_per_char' not in config['config'] else config['config']['pixels_per_char']
        if 'IAM' in mng.name:
            body_labels, body_imgs = generate_iam_line_imgs(
                mng, nb_lines=nb_body_lines, field_type="paragraphe", max_len=max_width_body//pixels_per_char, unique_font_path=unique_font_path, unique_size=unique_size
            )
        elif 'MULTI' in mng.name:
            body_labels, body_imgs = generate_iam_line_imgs(
                mng, nb_lines=nb_body_lines, field_type="paragraphe", max_len=max_width_body//pixels_per_char, multi=True, selected_dataset_name=selected_dataset_name,
                unique_font_path=unique_font_path, unique_size=unique_size
            )
        else:
            body_labels, body_imgs = generate_iam_mpopp_line_imgs(
                mng, nb_lines=nb_body_lines, field_type="paragraphe", min_len=8, max_len=max_width_body//pixels_per_char
            )

        ##############################################
        ## resize the generated images if needed: #################################
        same_resize = bool(unique_font_path and unique_size)
        body_imgs, _ = resize_img_widths(body_imgs, max_width_body, same_resize=same_resize)

        ###########################################################################################
        # apply the images from the body to the background starting from the top, until there are no more lines to place
        # to place or there is no more room in the allocated space
        body_lowest_point = area_from_the_top
        body_height = 0  # total height of the body
        i_body = 0  # number of lines added to the body
        body_leftest_point = area_left

        for img in body_imgs:
            remaining_height_body = area_bottom - body_lowest_point
            if img.shape[0] > remaining_height_body:
                nb_lines = 0
                break  # there is no space left in the body to add more lines

            background[
                body_lowest_point : body_lowest_point + img.shape[0],
                body_leftest_point : body_leftest_point + img.shape[1],
            ] = img
            body_height += img.shape[0]
            body_lowest_point += img.shape[0]
            nb_lines -= 1
            i_body += 1

        # body_top: after that, body_top becomes the lowest point of the body
        #####################################################################################

        area_from_the_top = body_lowest_point + randint(10, 50)

        ann_full_labels =  {key: '' for key in ['raw', 'begin','sem']}

        ###############################################
        marge_noms_full_labels = {key: '' for key in ['raw', 'begin','sem']}

        ###############################################
        body_full_labels = create_iam_labels(i_body, 'record', body_labels)

        section_labels = create_section_labels(matching_token, marge_noms_full_labels, ann_full_labels, body_full_labels, record_key='')
        # add the labels of the generated section to the labels of the page:
        for key in page_labels:
            if len(page_labels[key]) > 1:
                page_labels[key] += '\n'
            page_labels[key] += section_labels[key]

        ind_pg += 1

        if selected_dataset_name in ['IAM','IAM_NER']:
            if mng.params['config']['synthetic_data']['config'].get('iam',None) and mng.params['config']['synthetic_data']['config']['iam'].get('single_pg',False):
                break

    if crop:  # crop the area of the image under the last line (area_top was modified)
        background = background[:area_from_the_top]

    if 'MULTI' in mng.name or (mng.params['config'].get('layout_tokens_mode', '') == 'MULTI' and mng.params['config'].get('multi_layout_monolingual', False)):
        page_labels["sem"] += matching_token["ⓖ"]
    else:
        page_labels["sem"] += matching_token["ⓟ"]  # close the page

    for key in page_labels:
        page_labels[key] = page_labels[key].strip()
        # remove leading and trailing spaces

    return [background, page_labels, 2 if two_column else 1]

def get_iam_line_image(mng, field_type='paragraphe', min_len=0, max_len=10,alinea=False, last_line=False, no_pg=False):
    while True:
        sample = mng.samples[randint(0, len(mng))]
        for page in sample["pages_label"]:
            if no_pg:
                lines = page['text'].split('\n')
                random.shuffle(lines)
                for line in lines:
                    if len(line) > min_len and len(line) < max_len:
                        label = line.replace('ⓟ','').replace('ⓡ','').replace('Ⓟ','')
                        img = mng.generate_typed_text_line_image(label, field_type=field_type,alinea=alinea)
                        return label, img
            else:
                paragraphs = [
                    region_dict
                    for region_dict in page["paragraphs"]
                ]
                random.shuffle(paragraphs)
                for pg in paragraphs:
                    random.shuffle(pg["lines"])
                    for line in pg["lines"]:
                        if len(line["text"]) > min_len and len(line["text"]) < max_len:
                            label = line["text"].replace('ⓟ','').replace('ⓡ','')
                            if last_line: # the last line of a paragraph
                                nb_words = len(label.split(" ")) # number of words in the line
                                min_words = 1 if len(label.split(" ")[0])>=4 else 2 # we want at least 4 characters in the line or 2 words
                                new_nb_words = randint(min_words, nb_words+1)
                                label = ' '.join(label.split(" ")[:new_nb_words])
                            img = mng.generate_typed_text_line_image(label, field_type=field_type,alinea=alinea)
                            return label, img