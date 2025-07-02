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
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.dirname(PARENT_DIR))
sys.path.append(os.path.dirname(os.path.dirname(PARENT_DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(PARENT_DIR))))
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from Datasets.dataset_formatters.generic_dataset_formatter import \
    OCRDatasetFormatter

# Layout begin-token to end-token
SEM_MATCHING_TOKENS = {
            "ⓑ": "Ⓑ",  # paragraph (body)
            "ⓐ": "Ⓐ",  # annotation
            "ⓟ": "Ⓟ",  # page
            "ⓝ": "Ⓝ",  # page number
            "ⓢ": "Ⓢ",  # section (=linked annotation + body)
        }

READ_UNSUPPORTED_CHARS = ['1', 'ȳ', '̄']
class READ2016DatasetFormatter(OCRDatasetFormatter):
    def __init__(self, level, set_names=["train", "valid", "test"], dpi=150, end_token=True, sem_token=True):
        super(READ2016DatasetFormatter, self).__init__("READ_2016", level, "_sem" if sem_token else "", set_names)

        self.map_datasets_files.update({
            "READ_2016": {
                # (350 for train, 50 for validation and 50 for test)
                "page": {
                    "arx_files": ["Test-ICFHR-2016.tgz", "Train-And-Val-ICFHR-2016.tgz"],
                    "needed_files": [],
                    "format_function": self.format_read2016_page,
                },
                # (169 for train, 24 for validation and 24 for test)
                "double_page": {
                    "arx_files": ["Test-ICFHR-2016.tgz", "Train-And-Val-ICFHR-2016.tgz"],
                    "needed_files": [],
                    "format_function": self.format_read2016_double_page,
                }
            }
        })
        self.dpi = dpi
        self.end_token = end_token
        self.sem_token = sem_token
        self.matching_token = SEM_MATCHING_TOKENS

    def init_format(self):
        super().init_format()
        os.rename(os.path.join(self.temp_fold, "PublicData", "Training"), os.path.join(self.temp_fold, "train"))
        os.rename(os.path.join(self.temp_fold, "PublicData", "Validation"), os.path.join(self.temp_fold, "valid"))
        os.rename(os.path.join(self.temp_fold, "Test-ICFHR-2016"), os.path.join(self.temp_fold, "test"))
        os.rmdir(os.path.join(self.temp_fold, "PublicData"))
        for set_name in ["train", "valid", ]:
            for filename in os.listdir(os.path.join(self.temp_fold, set_name, "Images")):
                filepath = os.path.join(self.temp_fold, set_name, "Images", filename)
                if os.path.isfile(filepath):
                    os.rename(filepath, os.path.join(self.temp_fold, set_name, filename))
            os.rmdir(os.path.join(self.temp_fold, set_name, "Images"))

    def preformat_read2016(self):
        """
        Extract all information from READ 2016 dataset and correct some mistakes
        """
        def coord_str_to_points(coord_str):
            """
            Extract bounding box from coord string
            """
            points = coord_str.split(" ")
            x_points, y_points = list(), list()
            for p in points:
                y_points.append(int(p.split(",")[1]))
                x_points.append(int(p.split(",")[0]))
            top, bottom, left, right = np.min(y_points), np.max(y_points), np.min(x_points), np.max(x_points)
            return {
                "left": left,
                "bottom": bottom,
                "top": top,
                "right": right
            }

        def baseline_str_to_points(coord_str):
            """
            Extract bounding box from baseline string
            """
            points = coord_str.split(" ")
            x_points, y_points = list(), list()
            for p in points:
                y_points.append(int(p.split(",")[1]))
                x_points.append(int(p.split(",")[0]))
            top, bottom, left, right = np.min(y_points), np.max(y_points), np.min(x_points), np.max(x_points)
            return {
                "left": left,
                "bottom": bottom,
                "top": top,
                "right": right
            }

        dataset = {
            "train": list(),
            "valid": list(),
            "test": list(),
        }
        for set_name in ["train", "valid", "test"]:
            img_fold_path = os.path.join(self.temp_fold, set_name)
            xml_fold_path = os.path.join(self.temp_fold, set_name, "page")
            for xml_file_name in sorted(os.listdir(xml_fold_path)):
                if xml_file_name.split(".")[-1] != "xml":
                    continue
                filename = xml_file_name.split(".")[0]
                img_path = os.path.join(img_fold_path, filename + ".JPG")
                xml_file_path = os.path.join(xml_fold_path, xml_file_name)
                xml_root = ET.parse(xml_file_path).getroot()
                pages = xml_root.findall("{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Page")
                for page in pages:
                    page_dict = {
                        "label": list(),
                        "text_regions": list(),
                        "img_path": img_path,
                        "width": int(page.attrib["imageWidth"]),
                        "height": int(page.attrib["imageHeight"])
                    }
                    text_regions = page.findall("{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextRegion")
                    for text_region in text_regions:
                        text_region_dict = {
                            "label": list(),
                            "lines": list(),
                            "coords": coord_str_to_points(text_region.find("{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Coords").attrib["points"])
                        }
                        text_lines = text_region.findall("{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextLine")
                        for text_line in text_lines:
                            text_line_label = text_line.find("{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextEquiv")\
                                .find("{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Unicode")\
                                .text
                            if text_line_label is None and \
                                    text_line.attrib["id"] not in ["line_a5f4ab4e-2ea0-4c65-840c-4a89b04bd477",
                                                                   "line_e1288df8-8a0d-40df-be91-4b4a332027ec",
                                                                   "line_455330f3-9e27-4340-ae86-9d6c448dc091",
                                                                   "line_ecbbccee-e8c2-495d-ac47-0aff93f3d9ac",
                                                                   "line_e918616d-64f8-43d2-869c-f687726212be",
                                                                   "line_ebd8f850-1da5-45b1-b59c-9349497ecc8e",
                                                                   "line_816fb2ce-06b0-4e00-bb28-10c8b9c367f2"]:
                                print("ignored null line{}".format(page_dict["img_path"]))
                                continue
                            if text_line.attrib["id"] == "line_816fb2ce-06b0-4e00-bb28-10c8b9c367f2":
                                label = "16"
                            elif text_line.attrib["id"] == "line_a5f4ab4e-2ea0-4c65-840c-4a89b04bd477":
                                label = "108"
                            elif text_line.attrib["id"] == "line_e1288df8-8a0d-40df-be91-4b4a332027ec":
                                label = "196"
                            elif text_line.attrib["id"] == "line_455330f3-9e27-4340-ae86-9d6c448dc091":
                                label = "199"
                            elif text_line.attrib["id"] == "line_ecbbccee-e8c2-495d-ac47-0aff93f3d9ac":
                                label = "202"
                            elif text_line.attrib["id"] == "line_e918616d-64f8-43d2-869c-f687726212be":
                                label = "214"
                            elif text_line.attrib["id"] == "line_ebd8f850-1da5-45b1-b59c-9349497ecc8e":
                                label = "216"
                            else:
                                label = self.format_text_label(text_line_label)
                            line_baseline = text_line.find("{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Baseline")
                            line_coord = coord_str_to_points(text_line.find("{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Coords").attrib["points"])
                            text_line_dict = {
                                "label": label,
                                "coords": line_coord,
                                "baseline_coords": baseline_str_to_points(line_baseline.attrib["points"]) if line_baseline is not None else line_coord
                            }
                            text_line_dict["label"] = text_line_dict["label"]
                            text_region_dict["label"].append(text_line_dict["label"])
                            text_region_dict["lines"].append(text_line_dict)
                        if text_region_dict["label"] == list():
                            print("ignored null region {}".format(page_dict["img_path"]))
                            continue
                        text_region_dict["label"] = self.format_text_label("\n".join(text_region_dict["label"]))
                        text_region_dict["baseline_coords"] = {
                            "left": min([line["baseline_coords"]["left"] for line in text_region_dict["lines"]]),
                            "right": max([line["baseline_coords"]["right"] for line in text_region_dict["lines"]]),
                            "bottom": max([line["baseline_coords"]["bottom"] for line in text_region_dict["lines"]]),
                            "top": min([line["baseline_coords"]["top"] for line in text_region_dict["lines"]]),
                        }
                        page_dict["label"].append(text_region_dict["label"])
                        page_dict["text_regions"].append(text_region_dict)
                    page_dict["label"] = self.format_text_label("\n".join(page_dict["label"]))
                    dataset[set_name].append(page_dict)

        return dataset

    def format_read2016_page(self):
        """
        Format the READ 2016 dataset at single-page level
        """
        dataset = self.preformat_read2016()
        for set_name in ["train", "valid", "test"]:
            for i, page in enumerate(dataset[set_name]):
                new_img_name = "{}_{}.jpeg".format(set_name, i)
                new_img_path = os.path.join(self.target_fold_path, set_name, new_img_name)
                self.load_resize_save(page["img_path"], new_img_path, 300, self.dpi)
                new_label, sorted_text_regions, nb_cols, side = self.sort_text_regions(page["text_regions"], page["width"])
                paragraphs = list()
                for paragraph in page["text_regions"]:
                    paragraph_label = {
                        "label": paragraph["label"],
                        "lines": list(),
                        "mode": paragraph["mode"]
                    }
                    for line in paragraph["lines"]:
                        paragraph_label["lines"].append({
                            "text": line["label"],
                            "top": line["coords"]["top"],
                            "bottom": line["coords"]["bottom"],
                            "left": line["coords"]["left"],
                            "right": line["coords"]["right"],
                        })
                        paragraph_label["lines"][-1] = self.adjust_coord_ratio(paragraph_label["lines"][-1], self.dpi / 300)
                    paragraph_label["top"] = min([line["top"] for line in paragraph_label["lines"]])
                    paragraph_label["bottom"] = max([line["bottom"] for line in paragraph_label["lines"]])
                    paragraph_label["left"] = min([line["left"] for line in paragraph_label["lines"]])
                    paragraph_label["right"] = max([line["right"] for line in paragraph_label["lines"]])
                    paragraphs.append(paragraph_label)

                if self.sem_token:
                    if self.end_token:
                        new_label = "ⓟ" + new_label + "Ⓟ"
                    else:
                        new_label = "ⓟ" + new_label

                page_label = {
                    "text": new_label,
                    "paragraphs": paragraphs,
                    "nb_cols": nb_cols,
                    "side": side,
                    "top": min([pg["top"] for pg in paragraphs]),
                    "bottom": max([pg["bottom"] for pg in paragraphs]),
                    "left": min([pg["left"] for pg in paragraphs]),
                    "right": max([pg["right"] for pg in paragraphs]),
                    "page_width": int(np.array(Image.open(page["img_path"])).shape[1] * self.dpi / 300)
                }

                self.gt[set_name][new_img_name] = {
                    "text": new_label,
                    "nb_cols": nb_cols,
                    "pages": [page_label, ],
                }
                self.charset = self.charset.union(set(page["label"]))
        self.add_tokens_in_charset()

    def format_read2016_double_page(self):
        """
        Format the READ 2016 dataset at double-page level
        """
        dataset = self.preformat_read2016()
        for set_name in ["train", "valid", "test"]:
            for i, page in enumerate(dataset[set_name]):
                dataset[set_name][i]["label"], dataset[set_name][i]["text_regions"], dataset[set_name][i]["nb_cols"], dataset[set_name][i]["side"] = \
                    self.sort_text_regions(dataset[set_name][i]["text_regions"], dataset[set_name][i]["width"])
        dataset = self.group_by_page_number(dataset)
        for set_name in ["train", "valid", "test"]:
            i = 0
            for document in dataset[set_name]:
                if len(document["pages"]) != 2:
                    continue
                new_img_name = "{}_{}.jpeg".format(set_name, i)
                new_img_path = os.path.join(self.target_fold_path, set_name, new_img_name)
                img_left = np.array(Image.open(document["pages"][0]["img_path"]))
                img_right = np.array(Image.open(document["pages"][1]["img_path"]))
                left_page_width = img_left.shape[1]
                right_page_width = img_right.shape[1]
                img = np.concatenate([img_left, img_right], axis=1)
                img = self.resize(img, 300, self.dpi)
                img = Image.fromarray(img)
                img.save(new_img_path)
                pages = list()
                for page_id, page in enumerate(document["pages"]):
                    page_label = {
                        "text": page["label"],
                        "paragraphs": list(),
                        "nb_cols": page["nb_cols"]
                    }
                    for paragraph in page["text_regions"]:
                        paragraph_label = {
                            "label": paragraph["label"],
                            "lines": list(),
                            "mode": paragraph["mode"]
                        }
                        for line in paragraph["lines"]:
                            paragraph_label["lines"].append({
                                "text": line["label"],
                                "top": line["coords"]["top"],
                                "bottom": line["coords"]["bottom"],
                                "left": line["coords"]["left"] if page_id == 0 else line["coords"]["left"] + left_page_width,
                                "right": line["coords"]["right"]if page_id == 0 else line["coords"]["right"] + left_page_width,
                            })
                            paragraph_label["lines"][-1] = self.adjust_coord_ratio(paragraph_label["lines"][-1], self.dpi / 300)
                        paragraph_label["top"] = min([line["top"] for line in paragraph_label["lines"]])
                        paragraph_label["bottom"] = max([line["bottom"] for line in paragraph_label["lines"]])
                        paragraph_label["left"] = min([line["left"] for line in paragraph_label["lines"]])
                        paragraph_label["right"] = max([line["right"] for line in paragraph_label["lines"]])
                        page_label["paragraphs"].append(paragraph_label)
                    page_label["top"] = min([pg["top"] for pg in page_label["paragraphs"]])
                    page_label["bottom"] = max([pg["bottom"] for pg in page_label["paragraphs"]])
                    page_label["left"] = min([pg["left"] for pg in page_label["paragraphs"]])
                    page_label["right"] = max([pg["right"] for pg in page_label["paragraphs"]])
                    page_label["page_width"] = int(left_page_width * self.dpi / 300) if page_id == 0 else int(right_page_width * self.dpi / 300)
                    page_label["side"] = page["side"]
                    pages.append(page_label)

                label_left = document["pages"][0]["label"]
                label_right = document["pages"][1]["label"]
                if self.sem_token:
                    if self.end_token:
                        document_label = "ⓟ" + label_left + "Ⓟ" + "ⓟ" + label_right + "Ⓟ"
                    else:
                        document_label = "ⓟ" + label_left + "ⓟ" + label_right
                else:
                    document_label = label_left + "\n" + label_right
                self.gt[set_name][new_img_name] = {
                    "text": document_label,
                    "nb_cols": document["pages"][0]["nb_cols"] + document["pages"][1]["nb_cols"],
                    "pages": pages,
                }
                self.charset = self.charset.union(set(document_label))
                i += 1
        self.add_tokens_in_charset()

    def add_tokens_in_charset(self):
        """
        Add layout tokens to the charset
        """
        if self.sem_token:
            if self.end_token:
                self.charset = self.charset.union(set("ⓢⓑⓐⓝⓈⒷⒶⓃⓟⓅ"))
            else:
                self.charset = self.charset.union(set("ⓢⓑⓐⓝⓟ"))

    def group_by_page_number(self, dataset):
        """
        Group page data by pairs of successive pages
        """
        new_dataset = {
            "train": dict(),
            "valid": dict(),
            "test": dict()
        }
        for set_name in ["train", "valid", "test"]:
            for page in dataset[set_name]:
                page_num = int(page["text_regions"][0]["label"].replace(".", "").replace(" ", "").replace("ⓝ", "").replace("Ⓝ", ""))
                page["page_num"] = page_num
                if page_num in new_dataset[set_name]:
                    new_dataset[set_name][page_num].append(page)
                else:
                    new_dataset[set_name][page_num] = [page, ]
            new_dataset[set_name] = [{"pages": new_dataset[set_name][key],
                                      "page_num": new_dataset[set_name][key][0]["page_num"]
                                      } for key in new_dataset[set_name]]
        return new_dataset

    def update_label(self, label, start_token):
        """
        Add layout token to text region transcription
        """
        if self.sem_token:
            if self.end_token:
                return start_token + label + self.matching_token[start_token]
            else:
                return start_token + label
        return label

    def merge_group_tr(self, group, text_region):
        group["text_regions"].append(text_region)
        group["coords"]["top"] = min([tr["coords"]["top"] for tr in group["text_regions"]])
        group["coords"]["bottom"] = max([tr["coords"]["bottom"] for tr in group["text_regions"]])
        group["coords"]["left"] = min([tr["coords"]["left"] for tr in group["text_regions"]])
        group["coords"]["right"] = max([tr["coords"]["right"] for tr in group["text_regions"]])
        group["baseline_coords"]["top"] = min([tr["baseline_coords"]["top"] for tr in group["text_regions"]])
        group["baseline_coords"]["bottom"] = max([tr["baseline_coords"]["bottom"] for tr in group["text_regions"]])
        group["baseline_coords"]["left"] = min([tr["baseline_coords"]["left"] for tr in group["text_regions"]])
        group["baseline_coords"]["right"] = max([tr["baseline_coords"]["right"] for tr in group["text_regions"]])

    def is_annotation_alone(self, groups, page_width):
        for group in groups:
            if all([tr["coords"]["right"] < page_width / 2 and not tr["label"].replace(".", "").replace(" ", "").isdigit() for tr in group["text_regions"]]):
                return True
        return False

    def sort_text_regions(self, text_regions, page_width):
        """
        Establish reading order based on paragraph pixel position:
        page number then section by section: first all annotations, then associated body
        """
        nb_cols = 1
        groups = list()
        for text_region in text_regions:
            added_in_group = False
            temp_label = text_region["label"].replace(".", "").replace(" ", "")
            if len(temp_label) <= 4 and temp_label.isdigit():
                groups.append({
                    "coords": text_region["coords"].copy(),
                    "baseline_coords": text_region["baseline_coords"].copy(),
                    "text_regions": [text_region, ]
                })
                groups[-1]["coords"]["top"] = 0
                groups[-1]["coords"]["bottom"] = 0
                groups[-1]["baseline_coords"]["top"] = 0
                groups[-1]["baseline_coords"]["bottom"] = 0
                continue
            for group in groups:
                if not (group["baseline_coords"]["bottom"] <= text_region["baseline_coords"]["top"] or
                        group["baseline_coords"]["top"] >= text_region["baseline_coords"]["bottom"] or
                        (text_region["coords"]["right"]-text_region["coords"]["left"] > 0.4*page_width and
                         group["coords"]["right"]-group["coords"]["left"] > 0.4*page_width)):
                    self.merge_group_tr(group, text_region)
                    added_in_group = True
                    break

            if not added_in_group:
                groups.append({
                    "coords": text_region["coords"].copy(),
                    "baseline_coords": text_region["baseline_coords"].copy(),
                    "text_regions": [text_region, ]
                })
        while self.is_annotation_alone(groups, page_width):
            new_groups = list()
            annotations_groups = list()
            body_groups = list()
            for i in range(len(groups)):
                group = groups[i]
                if all([tr["label"].replace(".", "").replace(" ","").isdigit() for tr in group["text_regions"]]):
                    new_groups.append(group)
                    continue
                if all([tr["coords"]["right"] < page_width / 2 for tr in group["text_regions"]]):
                    annotations_groups.append(group)
                    continue
                body_groups.append(group)
            for ag in annotations_groups:
                dist = [min([abs(ag["coords"]["bottom"]-g["coords"]["top"]), abs(ag["coords"]["top"]-g["coords"]["bottom"])]) for g in body_groups]
                index = np.argmin(dist)
                for tr in ag["text_regions"]:
                    self.merge_group_tr(body_groups[index], tr)
            new_groups.extend(body_groups)
            groups = new_groups
        ordered_groups = sorted(groups, key=lambda g: g["coords"]["top"])
        sorted_text_regions = list()
        for group in ordered_groups:
            text_regions = group["text_regions"]
            if len(text_regions) == 1 and text_regions[0]["label"].replace(".", "").replace(" ", "").isdigit():
                side = "right" if text_regions[0]["coords"]["left"] > page_width / 2 else "left"
                sorted_text_regions.append(text_regions[0])
                sorted_text_regions[-1]["mode"] = "page_number"
                sorted_text_regions[-1]["label"] = self.update_label(sorted_text_regions[-1]["label"], "ⓝ")
            else:
                left = [tr for tr in group["text_regions"] if tr["coords"]["right"] < page_width / 2]
                right = [tr for tr in group["text_regions"] if tr["coords"]["right"] >= page_width / 2]
                nb_cols = max(2 if len(left) > 0 else 1, nb_cols)
                for i, text_region in enumerate(sorted(left, key=lambda tr: tr["coords"]["top"])):
                    sorted_text_regions.append(text_region)
                    sorted_text_regions[-1]["mode"] = "annotation"
                    sorted_text_regions[-1]["label"] = self.update_label(sorted_text_regions[-1]["label"], "ⓐ")
                    if i == 0 and self.sem_token:
                        sorted_text_regions[-1]["label"] = "ⓢ" + sorted_text_regions[-1]["label"]
                for i, text_region in enumerate(sorted(right, key=lambda tr: tr["coords"]["top"])):
                    sorted_text_regions.append(text_region)
                    sorted_text_regions[-1]["mode"] = "body"
                    sorted_text_regions[-1]["label"] = self.update_label(sorted_text_regions[-1]["label"], "ⓑ")
                    if i == 0 and self.sem_token and len(left) == 0:
                        sorted_text_regions[-1]["label"] = "ⓢ" + sorted_text_regions[-1]["label"]
                    if i == len(right)-1 and self.sem_token and self.end_token:
                        sorted_text_regions[-1]["label"] = sorted_text_regions[-1]["label"] + self.matching_token["ⓢ"]

        sep = "" if self.sem_token else "\n"
        new_label = sep.join(t["label"] for t in sorted_text_regions)
        return new_label, sorted_text_regions, nb_cols, side


if __name__ == "__main__":
    READ2016DatasetFormatter("page", sem_token=True).format()