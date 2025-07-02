#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
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

import os
import numpy as np
import xml.etree.ElementTree as ET
import re
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.dirname(PARENT_DIR))
sys.path.append(os.path.dirname(os.path.dirname(PARENT_DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(PARENT_DIR))))

from Datasets.dataset_formatters.generic_dataset_formatter import OCRDatasetFormatter
from Datasets.dataset_formatters.utils_dataset import natural_sort

# Layout string to token
SEM_MATCHING_TOKENS_STR = {
            'Ouverture': "ⓞ",  # opening
            'Corps de texte': "ⓑ",  # body
            'PS/PJ': "ⓟ",  # post scriptum
            'Coordonnées Expéditeur': "ⓢ",  # sender
            'Reference': "ⓢ",  # also counted as sender information
            'Objet': "ⓨ",  # why
            'Date, Lieu': "ⓦ",  # where, when
            'Coordonnées Destinataire': "ⓡ",  # recipient
        }

# Layout begin-token to end-token
SEM_MATCHING_TOKENS = {
            "ⓑ": "Ⓑ",
            "ⓞ": "Ⓞ",
            "ⓡ": "Ⓡ",
            "ⓢ": "Ⓢ",
            "ⓦ": "Ⓦ",
            "ⓨ": "Ⓨ",
            "ⓟ": "Ⓟ"
        }


RIMES_UNSUPPORTED_CHARS = ['À', 'Â', 'Ä', 'Ç', 'È', 'É', 'Ê', 'Ë', 'Î', 'Ï', 'Ô', 'Ö', 'Ù', 'Û', 'Ü']

class RIMESDatasetFormatter(OCRDatasetFormatter):
    def __init__(self, level, set_names=["train", "valid", "test"], dpi=150, sem_token=True):
        super(RIMESDatasetFormatter, self).__init__("RIMES", level, "_sem" if sem_token else "", set_names)

        self.source_fold_path = os.path.join("Datasets/raw", "RIMES")
        self.dpi = dpi
        self.sem_token = sem_token
        self.map_datasets_files.update({
            "RIMES": {
                # (1,050 for train, 100 for validation and 100 for test)
                "page": {
                    "arx_files": ["RIMES_page.tar.gz", ],
                    "needed_files": [],
                    "format_function": self.format_rimes_page,
                },
            }
        })

        self.matching_tokens_str = SEM_MATCHING_TOKENS_STR
        self.matching_tokens = SEM_MATCHING_TOKENS
        self.ordering_function = order_text_regions

    def preformat_rimes_page(self):
        """
        Extract all information from dataset and correct some annotations
        """
        dataset = {
            "train": list(),
            "valid": list(),
            "test": list()
        }
        img_folder_path = os.path.join(self.temp_fold, "RIMES page", "Images")
        xml_folder_path = os.path.join(self.temp_fold, "RIMES page", "XML")
        xml_files = natural_sort([os.path.join(xml_folder_path, name) for name in os.listdir(xml_folder_path)])
        train_xml = xml_files[:1050]
        valid_xml = xml_files[1050:1150]
        test_xml = xml_files[1150:]
        for set_name, xml_files in zip(self.set_names, [train_xml, valid_xml, test_xml]):
            for i, xml_path in enumerate(xml_files):
                text_regions = list()
                root = ET.parse(xml_path).getroot()
                img_name = root.find("source").text
                if img_name == "01160_L.png":
                    text_regions.append({
                        "label": "LETTRE RECOMMANDEE\nAVEC ACCUSE DE RECEPTION",
                        "type": "",
                        "coords": {
                            "left": 88,
                            "right": 1364,
                            "top": 1224,
                            "bottom": 1448,
                        }
                    })
                for text_region in root.findall("box"):
                    type = text_region.find("type").text
                    label = text_region.find("text").text
                    if label is None or len(label.strip()) <= 0:
                        continue
                    if label == "Ref : QVLCP¨65":
                        label = label.replace("¨", "")
                    if img_name == "01094_L.png" and type == "Corps de texte":
                        label = "Suite à la tempête du 19.11.06, un\narbre est tombé sur mon toît et l'a endommagé.\nJe d'eplore une cinquantaine de tuiles à changer,\nune poutre à réparer et une gouttière à\nremplacer. Veuillez trouver ci-joint le devis\nde réparation. Merci de m'envoyer votre\nexpert le plus rapidement possible.\nEn esperant une réponse rapide de votre\npart, veuillez accepter, madame, monsieur,\nmes salutations distinguées."
                    elif img_name == "01111_L.png" and type == "Corps de texte":
                        label = "Je vous ai envoyé un courrier le 20 octobre 2006\nvous signalant un sinistre survenu dans ma\nmaison, un dégât des eaux consécutif aux\nfortes pluis.\nVous deviez envoyer un expert pour constater\nles dégâts. Personne n'est venu à ce jour\nJe vous prie donc de faire le nécessaire\nafin que les réparations nécessaires puissent\nêtre commencés.\nDans l'attente, veuillez agréer, Monsieur,\nmes sincères salutations"

                    label = self.convert_label_accent(label)
                    label = self.convert_label(label)
                    label = self.format_text_label(label)
                    coords = {
                        "left": int(text_region.attrib["top_left_x"]),
                        "right": int(text_region.attrib["bottom_right_x"]),
                        "top": int(text_region.attrib["top_left_y"]),
                        "bottom": int(text_region.attrib["bottom_right_y"]),
                    }
                    text_regions.append({
                        "label": label,
                        "type": type,
                        "coords": coords
                    })
                text_regions = self.ordering_function(text_regions)
                dataset[set_name].append({
                    "text_regions": text_regions,
                    "img_path": os.path.join(img_folder_path, img_name),
                    "label": "\n".join([tr["label"] for tr in text_regions]),
                    "sem_label": "".join([self.sem_label(tr["label"], tr["type"]) for tr in text_regions]),
                })
        return dataset

    def convert_label_accent(self, label):
        """
        Solve encoding issues
        """
        return label.replace("\\n", "\n").replace("<euro>", "€").replace(">euro>", "€").replace(">fligne>", " ")\
            .replace("Â¤", "¤").replace("Ã»", "û").replace("�", "").replace("ï¿©", "é").replace("Ã§", "ç")\
            .replace("Ã©", "é").replace("Ã´", "ô").replace(u'\xa0', " ").replace("Ã¨", "è").replace("Â°", "°")\
            .replace("Ã", "À").replace("Ã¬", "À").replace("Ãª", "ê").replace("Ã®", "î").replace("Ã¢", "â")\
            .replace("Â²", "²").replace("Ã¹", "ù").replace("Ã", "à").replace("¬", "€")

    def format_rimes_page(self):
        """
        Format RIMES page dataset
        """
        dataset = self.preformat_rimes_page()
        for set_name in self.set_names:
            fold = os.path.join(self.target_fold_path, set_name)
            for sample in dataset[set_name]:
                new_name = "{}_{}.png".format(set_name, len(os.listdir(fold)))
                new_img_path = os.path.join(fold, new_name)
                self.load_resize_save(sample["img_path"], new_img_path, 300, self.dpi)
                for tr in sample["text_regions"]:
                    tr["coords"] = self.adjust_coord_ratio(tr["coords"], self.dpi / 300)
                page = {
                    "text": sample["label"] if not self.sem_token else sample["sem_label"],
                    "paragraphs": sample["text_regions"],
                    "nb_cols": 1,
                }
                self.charset = self.charset.union(set(page["text"]))
                self.gt[set_name][new_name] = page

    def convert_label(self, label):
        """
        Some annotations presents many options for a given text part, always keep the first one only
        """
        if "¤" in label:
            label = re.sub('¤{([^¤]*)[/|]([^¤]*)}¤', r'\1', label, flags=re.DOTALL)
            label = re.sub('¤{([^¤]*)[/|]([^¤]*)[/|]([^¤]*)>', r'\1', label, flags=re.DOTALL)
            label = re.sub('¤([^¤]*)[/|]([^¤]*)¤', r'\1', label, flags=re.DOTALL)
            label = re.sub('¤{}¤([^¤]*)[/|]([^ ]*)', r'\1', label, flags=re.DOTALL)
            label = re.sub('¤{/([^¤]*)/([^ ]*)', r'\1', label, flags=re.DOTALL)
            label = re.sub('¤{([^¤]*)[/|]([^ ]*)', r'\1', label, flags=re.DOTALL)
            label = re.sub('([^¤]*)/(.*)[¤}{]+', r'\1', label, flags=re.DOTALL)
            label = re.sub('[¤}{]+([^¤}{]*)[¤}{]+', r'\1', label, flags=re.DOTALL)
            label = re.sub('¤([^¤]*)¤', r'\1', label, flags=re.DOTALL)
        label = re.sub('[ ]+', " ", label, flags=re.DOTALL)
        label = label.strip()
        return label

    def sem_label(self, label, type):
        """
        Add layout tokens
        """
        if type == "":
            return label
        begin_token = self.matching_tokens_str[type]
        end_token = self.matching_tokens[begin_token]
        return begin_token + label + end_token


def order_text_regions(text_regions):
    """
    Establish reading order based on text region pixel positions
    """
    sorted_text_regions = list()
    for tr in text_regions:
        added = False
        if len(sorted_text_regions) == 0:
            sorted_text_regions.append(tr)
            added = True
        else:
            for i, sorted_tr in enumerate(sorted_text_regions):
                tr_height = tr["coords"]["bottom"] - tr["coords"]["top"]
                sorted_tr_height = sorted_tr["coords"]["bottom"] - sorted_tr["coords"]["top"]

                tr_is_totally_above = tr["coords"]["bottom"] < sorted_tr["coords"]["top"]

                tr_is_top_above = tr["coords"]["top"] < sorted_tr["coords"]["top"]

                is_same_level = sorted_tr["coords"]["top"] <= tr["coords"]["bottom"] <= sorted_tr["coords"]["bottom"] or\
                                sorted_tr["coords"]["top"] <= tr["coords"]["top"] <= sorted_tr["coords"]["bottom"] or\
                                tr["coords"]["top"] <= sorted_tr["coords"]["bottom"] <= tr["coords"]["bottom"] or\
                                tr["coords"]["top"] <= sorted_tr["coords"]["top"] <= tr["coords"]["bottom"]

                vertical_shared_space = tr["coords"]["bottom"]-sorted_tr["coords"]["top"] if tr_is_top_above else sorted_tr["coords"]["bottom"]-tr["coords"]["top"]

                reach_same_level_limit = vertical_shared_space > 0.3*min(tr_height, sorted_tr_height)

                is_more_at_left = tr["coords"]["left"] < sorted_tr["coords"]["left"]

                equivalent_height = abs(tr_height-sorted_tr_height) < 0.3*min(tr_height, sorted_tr_height)

                is_middle_above_top = np.mean([tr["coords"]["top"], tr["coords"]["bottom"]]) < sorted_tr["coords"]["top"]

                if tr_is_totally_above or\
                    (is_same_level and equivalent_height and is_more_at_left and reach_same_level_limit) or\
                    (is_same_level and equivalent_height and tr_is_top_above and not reach_same_level_limit) or\
                    (is_same_level and not equivalent_height and is_middle_above_top):
                    sorted_text_regions.insert(i, tr)
                    added = True
                    break
        if not added:
            sorted_text_regions.append(tr)

    return sorted_text_regions


if __name__ == "__main__":

    RIMESDatasetFormatter("page", sem_token=True).format()
    RIMESDatasetFormatter("page", sem_token=False).format()
