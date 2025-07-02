# This file is under a custom Research Usage Only (RUO) license.
# Please refer to the license file LICENSE for more details.
import re
import edlib

from nerval.evaluate import compute_matches, get_labels_aligned, compute_scores
from nerval.parse import get_type_label, get_position_label

NOT_ENTITY_TAG = "O"
BEGINNING_POS = ["B", "S", "U"]

def parse_bio_for_daniel(lines) -> dict:
    """Parse a BIO file to get text content, character-level NE labels and entity types count.

    Input : path to a valid BIO file
    Output format : { "words": str, "labels": list; "entity_count" : { tag : int } }
    """
    words = []
    labels = []
    entity_count = {"All": 0}
    last_tag = None

    # Track nested entities infos
    in_nested_entity = False
    containing_tag = None
    lines = [line for line in lines if line.split(' ')[0]]

    for index, line in enumerate(lines):
        word, label = line.split(' ')
        # Preserve hyphens to avoid confusion with the hyphens added later during alignment
        word = word.replace("-", "§")
        words.append(word)

        tag = get_type_label(label)

        # Spaces will be added between words and have to get a label
        if index != 0:
            # If new word has same tag as previous, not new entity and in entity, continue entity
            if (
                last_tag == tag
                and get_position_label(label) not in BEGINNING_POS
                and tag != NOT_ENTITY_TAG
            ):
                labels.append(f"I-{last_tag}")

            # If new word begins a new entity of different type, check for nested entity to correctly tag the space
            elif (
                last_tag != tag
                and get_position_label(label) in BEGINNING_POS
                and tag != NOT_ENTITY_TAG
                and last_tag != NOT_ENTITY_TAG
            ):
                # Advance to next word with different label as current
                future_label = label
                while (
                    index < len(lines)
                    and future_label != NOT_ENTITY_TAG
                    and get_type_label(future_label) != last_tag
                ):
                    index += 1
                    if index < len(lines):
                        future_label = lines[index].split()[1]

                # Check for continuation of the original entity
                if (
                    index < len(lines)
                    and get_position_label(future_label) not in BEGINNING_POS
                    and get_type_label(future_label) == last_tag
                ):
                    labels.append(f"I-{last_tag}")
                    in_nested_entity = True
                    containing_tag = last_tag
                else:
                    labels.append(NOT_ENTITY_TAG)
                    in_nested_entity = False

            elif in_nested_entity:
                labels.append(f"I-{containing_tag}")

            else:
                labels.append(NOT_ENTITY_TAG)
                in_nested_entity = False

        # Add a tag for each letter in the word
        if get_position_label(label) in BEGINNING_POS:
            labels += [f"B-{tag}"] + [f"I-{tag}"] * (len(word) - 1)
        else:
            labels += [label] * len(word)

        # Count nb entity for each type
        if get_position_label(label) in BEGINNING_POS:
            entity_count[tag] = entity_count.get(tag, 0) + 1
            entity_count["All"] += 1

        last_tag = tag

    result = None

    if words:
        result = dict()
        result["words"] = " ".join(words)
        result["labels"] = labels
        result["entity_count"] = entity_count

        assert len(result["words"]) == len(result["labels"])

    else:
        result = dict()
        result["words"] = ""
        result["labels"] = []
        result["entity_count"] = {"All": 0}

    return result

def format_pred_for_f1(values, non_character_tokens, named_entities, ne_format='after', ne_dict=None):
    def format_page_text_for_f1(page_text, non_character_tokens, named_entities, ne_format='after', ne_dict=None):
        BIO_list = []

        no_ne_words = ["&","(",")",",",".","/",":",";","=","?"]

        layout_tokens = set(non_character_tokens+'ⓐⒶ')-set(named_entities)

        for char in layout_tokens:
            page_text = page_text.replace(char,' ')
        page_text = re.sub(' +', ' ', page_text).strip()

        page_text = page_text.replace('. ',' . ').replace(', ',' , ').replace('; ',' ; ').replace('\n',' ')

        page_text = re.sub(' +', ' ', page_text).strip()

        if ne_format == 'after-no-hierarchy_exopopp':
            for tag in set(named_entities):
                page_text = page_text.replace(' ' +tag, tag)

        if 'exopopp' in ne_format:
            if 'both-side' in ne_format:
                ne_regex_begin = f"(^[{''.join(list(ne_dict.keys()))}]" + "{1,5})"
                ne_regex_end = f"([{''.join(list(ne_dict.values()))}]" + "{1,5})$"
                resersed_ne_dict = {v:k for k,v in ne_dict.items()}
            elif 'after' in ne_format:
                ne_regex = f"([{named_entities}]" + "{1,5}$)"
            elif 'before' in ne_format:
                ne_regex = f"(^[{named_entities}]" + "{1,5})"

        current_named_entities = set()
        previous_ne_name = ''
        word_list = re.split(' |&|\(|\)|,|\.|\/|:|;|=|\?', page_text)

        for word in word_list:
            if word:
                if 'after' in ne_format:
                    if 'exopopp' in ne_format:
                        if re.findall(ne_regex, word):
                            ne_span = [elem for elem in re.finditer(ne_regex, word)][-1].span()
                            pure_text_word = word[:ne_span[0]]
                            ne_name = word[ne_span[0]:]

                            bioe_prefix = 'I' if ne_name == previous_ne_name else 'B'
                            BIO_list.append(pure_text_word + ' ' + bioe_prefix + '-' + ne_name)
                        else:
                            ne_name = 'O'
                            BIO_list.append(word + ' ' + ne_name)
                    else:
                        if word[-1] in named_entities:
                            if len(word) != 1:
                                if word[-2] in named_entities:
                                    pure_text_word = word[:-2]
                                    ne_name = word[-2:]
                                else:
                                    pure_text_word = word[:-1]
                                    ne_name = word[-1:]

                                bioe_prefix = 'I' if ne_name == previous_ne_name else 'B'
                                BIO_list.append(pure_text_word + ' ' + bioe_prefix + '-' + ne_name)
                            else:
                                ne_name = 'O'
                                BIO_list.append(word + ' ' + ne_name)
                        else:
                            ne_name = 'O'
                            BIO_list.append(word + ' ' + ne_name)
                elif 'before' in ne_format:
                    if 'exopopp' in ne_format:
                        if re.findall(ne_regex, word):
                            ne_span = [elem for elem in re.finditer(ne_regex, word)][-1].span()
                            pure_text_word = word[ne_span[1]:]
                            ne_name = word[:ne_span[1]]

                            bioe_prefix = 'I' if ne_name == previous_ne_name else 'B'
                            BIO_list.append(pure_text_word + ' ' + bioe_prefix + '-' + ne_name)
                        else:
                            ne_name = 'O'
                            BIO_list.append(word + ' ' + ne_name)
                    else:
                        if word[0] in named_entities:
                            if len(word) != 1:
                                if word[1] in named_entities:
                                    pure_text_word = word[2:]
                                    ne_name = word[:2]
                                else:
                                    pure_text_word = word[1:]
                                    ne_name = word[:1]

                                bioe_prefix = 'I' if ne_name == previous_ne_name else 'B'
                                BIO_list.append(pure_text_word + ' ' + bioe_prefix + '-' + ne_name)
                            else:
                                ne_name = 'O'
                                BIO_list.append(word + ' ' + ne_name)
                        else:
                            ne_name = 'O'
                            BIO_list.append(word + ' ' + ne_name)
                elif 'both-side' in ne_format:
                    if 'exopopp' in ne_format:
                        pure_text_word = word
                        if re.findall(ne_regex_begin, word):
                            ne_that_start_span = [elem for elem in re.finditer(ne_regex_begin, word)][-1].span()
                            ne_that_start = pure_text_word[:ne_that_start_span[1]]
                            pure_text_word = pure_text_word[ne_that_start_span[1]:]

                            for ne_tag in ne_that_start:
                                current_named_entities.add(ne_tag)

                        if re.findall(ne_regex_end, pure_text_word):
                            ne_that_end_span = [elem for elem in re.finditer(ne_regex_end, pure_text_word)][-1].span()
                            ne_that_end = list(pure_text_word[ne_that_end_span[0]:])
                            pure_text_word = pure_text_word[:ne_that_end_span[0]]

                        ne_name = ''.join(current_named_entities)
                        if ne_name and not word in no_ne_words:
                            bioe_prefix = 'I' if ne_name == previous_ne_name else 'B'
                            BIO_list.append(pure_text_word + ' ' + bioe_prefix + '-' + ne_name)
                        else:
                            BIO_list.append(word + ' O')

                        if re.findall(ne_regex_end, word):
                            for ne_tag in ne_that_end:
                                if resersed_ne_dict[ne_tag] in current_named_entities:
                                    current_named_entities.remove(resersed_ne_dict[ne_tag])
                else:
                    print('F1 metric error, no tagging type specified')

                previous_ne_name = ne_name

        return BIO_list

    results_dict = {
        values['names'][i] : {
            'gt':format_page_text_for_f1(values['str_y'][i], non_character_tokens=non_character_tokens, named_entities=named_entities, ne_format=ne_format, ne_dict=ne_dict),
            'pred':format_page_text_for_f1(values['str_x'][i], non_character_tokens=non_character_tokens, named_entities=named_entities, ne_format=ne_format, ne_dict=ne_dict)
        } for i in range(len(values['names']))
    }

    return results_dict

def compute_f1(ner_metrics_global_dict, threshold = 0.3):
    img_evals = []
    total_dict = {'P': 0, 'R': 0, 'F1': 0, 'Support': 0, 'predicted': 0, 'matched': 0}
    for img_dict in ner_metrics_global_dict.values():
        img_eval = daniel_eval(img_dict['gt'], img_dict['pred'], threshold)
        if img_eval['All']['Support'] or img_eval['All']['predicted']:
            for key in ['P','R','F1','predicted','matched', 'Support']:
                if img_eval['All'][key] is None:
                    img_eval['All'][key] = 0
            img_evals.append(img_eval)
            total_dict['P'] += img_eval['All']['P']
            total_dict['R'] += img_eval['All']['R']
            total_dict['F1'] += img_eval['All']['F1']
            total_dict['Support'] += img_eval['All']['Support']
            total_dict['predicted'] += img_eval['All']['predicted']
            total_dict['matched'] += img_eval['All']['matched']

    total_dict['P'] /= max(len(img_evals),1)
    total_dict['R'] /= max(len(img_evals),1)
    total_dict['F1'] /= max(len(img_evals),1)

    return img_evals, total_dict

def daniel_eval(annot_lines, predict_lines, threshold) -> dict:
    """Compute recall and precision for each entity type found in annotation and/or prediction.

    Each measure is given at document level, global score is a micro-average across entity types.
    """
    annot = parse_bio_for_daniel(annot_lines)
    predict = parse_bio_for_daniel(predict_lines)
    # Align annotation and prediction
    if not predict["words"]:
        predict["words"] = 'A'
        predict["labels"] = 'O'
    align_result = edlib.align(annot["words"], predict["words"], task="path")
    nice_alignment = edlib.getNiceAlignment(
        align_result, annot["words"], predict["words"]
    )

    annot_aligned = nice_alignment["query_aligned"]
    predict_aligned = nice_alignment["target_aligned"]

    # Align labels from string alignment
    labels_annot_aligned = get_labels_aligned(
        annot["words"], annot_aligned, annot["labels"]
    )
    labels_predict_aligned = get_labels_aligned(
        predict["words"], predict_aligned, predict["labels"]
    )

    # Get nb match
    matches = compute_matches(
        annot_aligned,
        predict_aligned,
        labels_annot_aligned,
        labels_predict_aligned,
        threshold,
    )

    # Compute scores
    scores = compute_scores(annot["entity_count"], predict["entity_count"], matches)

    return scores
